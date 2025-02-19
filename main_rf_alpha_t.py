import time
import os
from copy import deepcopy
from functools import partial
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker
from tqdm.auto import trange
import matplotlib.pyplot as plt

# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

from model_rf import (
    RectifiedFlow, ResidualNetwork, 
    get_timestep_embedding, get_flow_soln_kwargs, 
    velocity_to_score, score_to_velocity
)
from train_rf import make_step, apply_ema, cosine_time, identity
from utils import (
    ppca, get_opt_and_state, clear_and_get_results_dir, 
    create_gif, plot_losses, plot_samples, 
    clip_latents, get_data, measurement,
    exists, default, maybe_clip, maybe_invert
)
from soap import soap

typecheck = jaxtyped(typechecker=typechecker)

YArray = Float[Array, "2"]

YCovariance = Float[Array, "2 2"]

XArray = Float[Array, "2"]

XCovariance = Float[Array, "2 2"]

TCovariance = Float[Array, "2 2"]

XSampleFn = Callable[[PRNGKeyArray], XArray]

XYSampleFn = Callable[[PRNGKeyArray, XArray], XArray]
 
SDEType = Literal["non-singular", "zero-ends", "singular", "gamma"]


"""
    Posterior sampling 
"""


@typecheck
def get_score_y_x_cg(
    y_: YArray, 
    x: XArray, 
    t: Scalar, 
    flow: RectifiedFlow,
    cov_x: XCovariance,
    cov_y: YCovariance,
    *,
    max_iter: int = 5,
    tol: float = 1e-5, 
    return_score_x: bool = False
) -> XArray:

    cov_t = get_cov_t(flow, t)

    x, vjp = jax.vjp(lambda x_t: velocity_to_score(flow, t, x_t), x) # This shouldn't be score?

    y = x # If no A, x is E[x|x_t]?

    # Get linear operator Mv = b to solve for v given b, choosing heuristic for V[x|x_t]
    if cov_x is None:
        cov_y_xt = lambda v: cov_y @ v + cov_t * vjp(v) # Is this right?
    else:
        cov_x_xt = cov_t + (-(cov_t ** 2.)) * jnp.linalg.inv(cov_x + cov_t)
        cov_y_xt = lambda v: cov_y @ v + cov_x_xt @ v

    b = y_ - y # y_ is data

    # This is a linear operator in lineax?
    v, _ = jax.scipy.sparse.linalg.cg(
        A=cov_y_xt, b=b, tol=tol, maxiter=max_iter
    )

    (score,) = vjp(v) 

    if return_score_x:
        return x + cov_t @ score, score # divide by alpha_t
    else:
        return x + cov_t @ score


def value_and_jacfwd(f: Callable, x: XArray, **kwargs) -> tuple[XArray, XCovariance]:
    _fn = lambda x: f(x, **kwargs)
    J_fn = partial(jax.jvp, _fn, (x,)) # NOTE: J[E[x|x_t]] w.r.t. x_t 
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, J = jax.vmap(J_fn, out_axes=(None, 1))((basis,))
    return y, J


@typecheck
def get_E_x_x_t_gaussian(
    x_t: XArray, 
    alpha_t: Scalar,
    cov_t: TCovariance, 
    score_t: XArray
) -> XArray: 
    # Convert score to expectation via Tweedie; x_t + cov_t * score[p(x_t)]
    return (x_t + cov_t @ score_t) / maybe_clip(alpha_t)


@typecheck
def get_E_x_x_t(
    x_t: XArray,
    flow: RectifiedFlow,
    t: Scalar
) -> XArray: 
    # Get E[x|x_t] from RF E[x_1 - x_0|x_t]; Eq 55, Appendix B.1 (E[x|x_t]=x_t-tE[x_1|x_t])
    return x_t - t * flow.v(t, x_t)


@typecheck
def get_cov_t(flow: RectifiedFlow, t: Scalar) -> TCovariance:
    # Calculate the covariance of p(x_t|x) = G[x_t|alpha_t * x, Sigma_t] 
    (dim,) = flow.x_shape 
    var_t = maybe_clip(jnp.square(flow.sigma(t))) 
    cov_t = jnp.identity(dim) * var_t
    return cov_t


@typecheck
def get_score_y_x(
    y_: YArray, # Data
    x: XArray, # x_t
    t: Scalar, 
    flow: RectifiedFlow,
    cov_y: YCovariance,
    return_score_x: bool = False
) -> Union[XArray, tuple[XArray, XArray]]:
    # Score of Gaussian linear data-likelihood G[y|x, cov_y] 

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    score_x = velocity_to_score(flow, t, x)

    E_x_x_t, J_E_x_x_t = value_and_jacfwd(get_E_x_x_t, x, flow=flow, t=t)

    V_x_x_t = cov_t @ J_E_x_x_t / alpha_t 
    V_y_x_t = cov_y + V_x_x_t 

    score_y_x = J_E_x_x_t @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20, EMdiff

    if return_score_x:
        return score_y_x, score_x
    else:
        return score_y_x


@typecheck
def get_score_gaussian_y_x(
    y_: YArray, 
    x: XArray, # x_t
    t: Scalar,
    flow: RectifiedFlow,
    cov_y: YCovariance,
    mu_x: XArray,
    inv_cov_x: XCovariance 
) -> XArray:
    # NOTE: alpha_t correct here?

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_clip(alpha_t)  # Tweedie with score of analytic G[x|mu_x, cov_x]

    E_x_x_t, J_E_x_x_t = value_and_jacfwd(
        get_E_x_x_t_gaussian, x, alpha_t=alpha_t, cov_t=cov_t, score_t=score_x
    ) 

    V_x_x_t = cov_t @ J_E_x_x_t / maybe_clip(alpha_t) # Approximation to Eq 21, see Eq 22. (or heuristics; cov_t, inv(cov_t)...)
    V_y_x_t = cov_y + V_x_x_t
    
    score_y_x = J_E_x_x_t @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20

    return score_y_x


@typecheck
def get_score_gaussian_x_y(
    y_: YArray, 
    x: XArray, # x_t
    t: Scalar,
    flow: RectifiedFlow,
    cov_y: YCovariance,
    mu_x: XArray,
    inv_cov_x: XCovariance
) -> XArray:
    # Score of Gaussian kernel centred on data NOTE: mode="Gaussian" for init

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_clip(alpha_t) # Tweedie with score of analytic G[x|mu_x, cov_x]

    score_y_x = get_score_gaussian_y_x(
        y_, x, t, flow, cov_y, mu_x, inv_cov_x
    )

    score_x_y = score_y_x + score_x

    return score_x_y


"""
    DDIM
"""


@typecheck
@eqx.filter_jit
def single_x_y_ddim_sample_fn(
    flow: RectifiedFlow,
    key: PRNGKeyArray, 
    y_: YArray,
    cov_y: YCovariance, 
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[XCovariance] = None, 
    *,
    q_0_sampling: bool = False, # Sampling initially or not
    n_steps: int = 500,
    eta: float = 1., # DDIM stochasticity 
    sde_type: SDEType = "zero-ends",
    mode: Literal["full", "cg"] = "full"
) -> XArray:
    # DDIM sampler including data-likelihood score

    key_z, key_sample = jr.split(key)

    # Reversed times
    times = jnp.linspace(
        flow.soln_kwargs["t1"], 
        flow.soln_kwargs["t0"], 
        n_steps + 1 
    )

    flow = eqx.nn.inference_mode(flow)

    def sample_step(i: Scalar, x_t_key: XArray) -> tuple[XArray, PRNGKeyArray]:
        x_t, key = x_t_key

        s, t = times[i], times[i + 1]
        dt = t - s

        if q_0_sampling:
            assert (mu_x is not None) and (inv_cov_x is not None)

            # Implement CG method for this
            score_y_x, score_x = get_score_gaussian_x_y(
                y_, x_t, t, flow, cov_y=cov_y, mu_x=mu_x, inv_cov_x=inv_cov_x
            )
        else:
            if mode == "full":
                score_y_x, score_x = get_score_y_x(
                    y_, x_t, t, flow, cov_y, return_score_x=True # x is x_t
                ) 
            if mode == "cg":
                score_y_x, score_x = get_score_y_x_cg(
                    y_, x_t, t, flow, cov_x, cov_y, return_score_x=True
                ) 

        score_x_y = score_y_x + score_x

        drift, diffusion = flow.sde(z, t, sde_type=sde_type) # NOTE: implement SDE here?! DDIM for arbitrary SDE => use zero ends here 

        # # Compute deterministic step for DDIM (reverse ODE)
        deterministic_update = drift - 0.5 * jnp.square(diffusion) * score_x_y

        # Stochasticity term (optional, eta=0 for DDIM)
        noise = jr.normal(key, x_t.shape)
        stochastic_update = eta * diffusion * jnp.sqrt(jnp.abs(dt)) * noise # NOTE: abs on dt?

        # Update x_t using DDIM 
        x_s = x_t + deterministic_update * dt + stochastic_update

        return x_s, key

    z = jr.normal(key_z, flow.x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, key_sample)
    )

    return x


"""
    ODE sampling
"""


@typecheck
@eqx.filter_jit
def single_sample_fn_ode(
    flow: RectifiedFlow, 
    key: PRNGKeyArray, 
    *,
    alpha: float = 0.1,
    sde_type: SDEType = "zero-ends"
) -> XArray:
    """ Sample ODE of non-singular SDE corresponding to Gaussian flow matching marginals """

    def ode(t: Scalar, x: XArray, args: Optional[tuple]) -> XArray:
        # Non-Singular ODE; using score of Gaussian Rectified Flow
        t = jnp.asarray(t)
        v = flow.v(t, x)

        # score = -((1. - t) * v + x) / t # Assuming mu_1, sigma_1 = 0, 1 #NOTE: bug! -x not +x
        # score = velocity_to_score(flow=None, t=t, x=x, velocity=v)
        # drift = v + 0.5 * jnp.square(alpha) * t * score # Non-singular SDE
        # return drift

        score = velocity_to_score(flow=None, t=t, x=x, velocity=v)
        # drift, diffusion = flow.sde(
        #     x, t, alpha=alpha, sde_type=sde_type
        # ) 
        # return drift - 0.5 * jnp.square(diffusion) * score # Posterior score
        return flow.reverse_ode(x, t, score, alpha=alpha, sde_type=sde_type)
    
    flow = eqx.nn.inference_mode(flow, True)

    z = jr.normal(key, flow.x_shape)

    term = dfx.ODETerm(ode) 

    sol = dfx.diffeqsolve(
        term, 
        flow.solver, 
        flow.t1, 
        flow.t0, 
        -flow.dt0, 
        z
    )

    (x_,) = sol.ys
    return x_


@typecheck
@eqx.filter_jit
def single_x_y_sample_fn_ode(
    flow: RectifiedFlow,
    key: PRNGKeyArray, 
    y_: YArray,
    cov_y: YCovariance, 
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[XCovariance] = None, 
    *,
    mode: Literal["full", "cg"] = "full",
    sde_type: SDEType = "zero-ends",
    q_0_sampling: bool = False # Sampling initial q_0(x) or model q(x)
) -> XArray:
    # Latent posterior sampling function

    def reverse_ode(t: Scalar, x: XArray, args: Optional[tuple]) -> XArray:
        # Sampling along conditional score p(x|y)
        t = jnp.asarray(t)

        if q_0_sampling:
            # Implement CG method for this
            score_y_x, score_x = get_score_gaussian_x_y(
                y_, x, t, flow, cov_y=cov_y, mu_x=mu_x, inv_cov_x=inv_cov_x
            )
        else:
            if mode == "full":
                score_y_x, score_x = get_score_y_x(
                    y_, x, t, flow, cov_y, return_score_x=True # x is x_t
                ) 
            if mode == "cg":
                score_y_x, score_x = get_score_y_x_cg(
                    y_, x, t, flow, cov_x, cov_y, return_score_x=True
                ) 

        score_x_y = score_x + score_y_x

        return flow.reverse_ode(x, t, score=score_x_y, sde_type=sde_type) 

    sol = dfx.diffeqsolve(
        dfx.ODETerm(reverse_ode), 
        **get_flow_soln_kwargs(flow, reverse=True),
        y0=jr.normal(key, flow.x_shape) 
    )
    return sol.ys[0]


"""
    Stochastic sampling
"""


@typecheck
@eqx.filter_jit
def single_non_singular_sample_fn(
    flow: RectifiedFlow, 
    key: PRNGKeyArray, 
    *,
    g_scale: float = 0.1, 
    n_steps: int = 500,
    n: float = 1., 
    m: float = 0.
) -> XArray:
    """
        Stochastic sampling of RF
    """

    key_z, key_sample = jr.split(key)

    t = jnp.linspace(
        flow.soln_kwargs["t1"], flow.soln_kwargs["t0"], n_steps + 1
    )

    flow = eqx.nn.inference_mode(flow)

    def sample_step(i, z):
        z, key = z

        key, key_eps = jr.split(key)

        _t = t[i] 
        _dt = t[i + 1] - t[i]

        eps = jr.normal(key_eps, z.shape)

        z_hat = flow.v(1. - _t, z) # Add velocity(score_y_x) here

        _z_hat = -z_hat
        g = g_scale * jnp.power(_t, 0.5 * n) * jnp.power(1. - _t, 0.5 * m)
        s_u = -((1. - _t) * _z_hat + z)
        fr = _z_hat - jnp.square(g_scale) * jnp.power(_t, n - 1.) * jnp.power(1. - _t, m) * 0.5 * s_u

        dbt = jnp.sqrt(jnp.abs(_dt)) * eps
        z = z + fr * _dt + g * dbt

        return z, key 

    z = jr.normal(key_z, flow.x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, key_sample)
    )

    return x


@typecheck
@eqx.filter_jit
def single_non_singular_x_y_sample_fn(
    flow: RectifiedFlow, 
    key: PRNGKeyArray, 
    y_: YArray,
    cov_y: YCovariance,
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[XCovariance] = None,
    *,
    g_scale: float = 0.1, 
    n_steps: int = 500,
    n: float = 1., 
    m: float = 0.,
    mode: Literal["full", "cg"] = "full",
    q_0_sampling: bool = False
) -> XArray:
    """
        Stochastic sampling of p(x|y)
    """

    key_z, key_sample = jr.split(key)

    # Reversed time
    t = jnp.linspace(
        flow.soln_kwargs["t1"], flow.soln_kwargs["t0"], n_steps + 1
    )

    flow = eqx.nn.inference_mode(flow)

    def sample_step(i, z):
        z, key = z

        key, key_eps = jr.split(key)

        _t = t[i] 
        _dt = t[i + 1] - t[i]

        eps = jr.normal(key_eps, z.shape)

        if q_0_sampling:
            assert inv_cov_x is not None

            score_p_y_x_t = get_score_gaussian_y_x(
                y_, z, 1. - _t, flow, cov_y=cov_y, mu_x=mu_x, inv_cov_x=inv_cov_x
            )
        else:
            if mode == "full":
                score_p_y_x_t = get_score_y_x(
                    y_, z, 1. - _t, flow, cov_y # x is x_t
                ) 
            if mode == "cg":
                score_p_y_x_t = get_score_y_x_cg(
                    y_, z, 1. - _t, flow, cov_x, cov_y
                ) 

        # Adding velocity[score(p(y|x))]
        # z_hat = flow.v(1. - _t, z) + score_to_velocity(score_p_y_x_t, 1. - _t, z) # Add velocity(score_y_x) here
        # Adding velocity[p(y|x)]
        z_hat = flow.v(1. - _t, z) + jnp.square(flow.sigma(1. - _t)) * (1. / (_t * (1. - _t))) * score_p_y_x_t # Add velocity(score_y_x) here

        _z_hat = -z_hat
        g = g_scale * jnp.power(_t, 0.5 * n) * jnp.power(1. - _t, 0.5 * m)
        s_u = -((1. - _t) * _z_hat + z)
        fr = _z_hat - jnp.square(g_scale) * jnp.power(_t, n - 1.) * jnp.power(1. - _t, m) * 0.5 * s_u

        dbt = jnp.sqrt(jnp.abs(_dt)) * eps
        z = z + fr * _dt + g * dbt

        return z, key 

    z = jr.normal(key_z, flow.x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, key_sample)
    )

    return x


"""
    Sampler utils
"""


@typecheck
def get_x_y_sampler_ddim(
    flow: RectifiedFlow, 
    cov_y: YCovariance, 
    mu_x: Optional[XArray] = None, 
    cov_x: Optional[XCovariance] = None, 
    *,
    n_steps: int = 500,
    eta: float = 1., # Stochasticity in DDIM
    mode: Literal["full", "cg"] = "full",
    sde_type: SDEType = "zero-ends",
    q_0_sampling: bool = False # Sampling initial q_0(x) or model q(x)
) -> XYSampleFn:

    inv_cov_x = maybe_invert(cov_x)

    fn = lambda key, y_: single_x_y_ddim_sample_fn(
        flow, 
        key, 
        y_, 
        cov_y=cov_y, 
        mu_x=mu_x,
        inv_cov_x=inv_cov_x, 
        q_0_sampling=q_0_sampling, 
        n_steps=n_steps,
        eta=eta,
        sde_type=sde_type,
        mode=mode # NOTE: Why does this need cov_x? Does cov_x need iterating?
    )
    return fn


@typecheck
def get_x_y_sampler_ode(
    flow: RectifiedFlow, 
    cov_y: YCovariance, 
    mu_x: Optional[XArray] = None, 
    cov_x: Optional[XCovariance] = None, 
    *,
    mode: Literal["full", "cg"] = "full",
    sde_type: SDEType = "zero-ends",
    q_0_sampling: bool = False
) -> XYSampleFn:

    inv_cov_x = maybe_invert(cov_x)

    fn = lambda key, y_: single_x_y_sample_fn_ode(
        flow, 
        key, 
        y_, 
        cov_y, 
        mu_x,
        inv_cov_x, 
        mode=mode, 
        sde_type=sde_type,
        q_0_sampling=q_0_sampling # NOTE: Why does this need cov_x? Does cov_x need iterating?
    )
    return fn


@typecheck
def get_x_y_sampler_sde(
    flow: RectifiedFlow, 
    cov_y: YCovariance, 
    mu_x: Optional[XArray] = None, 
    cov_x: Optional[XCovariance] = None, 
    *,
    n_steps: int = 500,
    g_scale: float = 0.1,
    n: float = 1.,
    m: float = 0.,
    mode: Literal["full", "cg"] = "full",
    q_0_sampling: bool = False # Sampling initial q_0(x) or model q(x)
) -> XYSampleFn:

    inv_cov_x = maybe_invert(cov_x)

    fn = lambda key, y_: single_non_singular_x_y_sample_fn(
        flow, 
        key, 
        y_, 
        cov_y=cov_y, 
        mu_x=mu_x,
        inv_cov_x=inv_cov_x, 
        n_steps=n_steps,
        g_scale=g_scale,
        n=n,
        m=m,
        mode=mode,
        q_0_sampling=q_0_sampling, 
    )
    return fn


@typecheck
def get_ode_sample_fn(
    flow: RectifiedFlow,
    *,
    alpha: float = 0.1,
    sde_type: SDEType = "zero-ends"
) -> XSampleFn:
    # Sampler for e.g. sampling latents from model p(x) without y 
    fn = lambda key: single_sample_fn_ode(
        flow, key=key, alpha=alpha, sde_type=sde_type
    )
    return fn


@typecheck
def get_non_singular_sample_fn(
    flow: RectifiedFlow,
    *,
    n_steps: int = 500,
    g_scale: float = 0.1,
    n: float = 1.,
    m: float = 0.
) -> XSampleFn:
    # Sampler for e.g. sampling latents from model p(x) without y 
    fn = lambda key: single_non_singular_sample_fn(
        flow, key=key, g_scale=g_scale, n_steps=n_steps, n=n, m=m
    )
    return fn


def test_train(key, flow, X, n_batch, diffusion_iterations, use_ema, save_dir):
    # Test whether training config fits FM model on latents

    opt, opt_state = get_opt_and_state(flow, soap, lr=1e-3)

    if use_ema:
        ema_flow = deepcopy(flow)

    losses_i = []
    with trange(
        diffusion_iterations, desc="Training (test)", colour="red"
    ) as steps:
        for i in steps:
            key_x, key_step = jr.split(jr.fold_in(key, i))

            x = jr.choice(key_x, X, (n_batch,)) # Make sure always choosing x ~ p(x|y)

            L, flow, key, opt_state = make_step(
                flow, 
                x, 
                key_step, 
                opt_state, 
                opt.update, 
                loss_type="mse", 
                time_schedule=identity
            )

            if use_ema:
                ema_flow = apply_ema(ema_flow, flow, ema_rate)

            losses_i.append(L)
            steps.set_postfix_str(f"{L=:.3E}")

    key, key_sample = jr.split(key)

    # Generate latents from q(x)
    keys = jr.split(key_sample, 8000)
    X_test_sde = jax.vmap(get_non_singular_sample_fn(flow))(keys)
    X_test_ode = jax.vmap(partial(single_sample_fn_ode, flow, sde_type=sde_type))(keys)

    plt.figure()
    plt.scatter(*X.T, s=0.1, color="k")
    plt.scatter(*X_test_sde.T, s=0.1, color="b")
    plt.scatter(*X_test_ode.T, s=0.1, color="r")
    plt.savefig(os.path.join(save_dir, "test.png"))
    plt.close()


if __name__ == "__main__":
    key = jr.key(int(time.time()))

    save_dir             = clear_and_get_results_dir(save_dir="imgs__/")

    # Train
    test_on_latents      = True
    em_iterations        = 64
    diffusion_iterations = 5_000
    n_batch              = 5_000
    loss_type            = "mse"
    time_schedule        = identity
    lr                   = 1e-3
    optimiser            = soap
    use_lr_schedule      = False #True
    initial_lr           = 1e-6
    n_epochs_warmup      = 1
    ppca_pretrain        = False
    n_pca_iterations     = 256
    clip_x_y             = False #True # Clip sampled latents
    x_clip_limit         = 4.
    re_init_opt_state    = True
    n_plot               = 10_000
    sampling_mode        = "ode"        # ODE, SDE or DDIM
    sde_type             = "zero-ends"  # SDE of flow ODE
    mode                 = "full"       # CG mode or not
    use_ema              = True
    ema_rate             = 0.9999

    # Model
    width_size           = 256 
    depth                = 3 
    activation           = jax.nn.gelu 
    soln_kwargs          = dict(t0=0., dt0=0.02, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    time_embedding_dim   = 64

    # Data
    data_dim             = 2
    n_data               = 100_000
    sigma_y              = 0.05 # Tiny eigenvalues may have been numerically unstable?
    cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)

    assert sampling_mode in ["ddim", "ode", "sde"]
    assert sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert mode in ["full", "cg"]

    key_net, key_data, key_measurement, key_ppca, key_em = jr.split(key, 5)

    # Rectified flow model
    time_embedder = partial(
        get_timestep_embedding, embedding_dim=time_embedding_dim
    )

    net = ResidualNetwork(
        data_dim, 
        width_size=width_size, 
        depth=depth, 
        t_embedding_dim=time_embedding_dim, 
        t1=soln_kwargs["t1"],
        key=key_net 
    )

    flow = RectifiedFlow(
        net, time_embedder, x_shape=(data_dim,), **soln_kwargs
    )

    if use_ema:
        ema_flow = deepcopy(flow)

    opt, opt_state = get_opt_and_state(
        flow, 
        optimiser,
        lr=lr, 
        use_lr_schedule=use_lr_schedule, 
        initial_lr=initial_lr,
        n_epochs_warmup=n_epochs_warmup,
        n_data=n_data,
        n_batch=n_batch,
        diffusion_iterations=diffusion_iterations
    )

    # Latents
    X = get_data(key_data, n_data)

    # Generate y ~ G[y|x, cov_y]
    keys = jr.split(key_measurement, n_data)
    Y = jax.vmap(partial(measurement, cov_y=cov_y))(keys, X) 

    if test_on_latents:
        test_train(
            key, 
            flow, 
            X, 
            n_batch=n_batch,
            diffusion_iterations=diffusion_iterations, 
            use_ema=use_ema,
            save_dir=save_dir
        )

    # PPCA pre-training for q_0(x|mu_x, cov_x)
    mu_x = jnp.zeros(data_dim)
    cov_x = jnp.identity(data_dim) # Start at cov_y?
    if ppca_pretrain:

        X_ = Y
        for s in trange(
            n_pca_iterations, desc="PPCA Training", colour="blue"
        ):
            key_pca, key_sample = jr.split(jr.fold_in(key_ppca, s))

            mu_x, cov_x = ppca(X_, key_pca, rank=data_dim)

            if sampling_mode == "ddim":
                sampler = get_x_y_sampler_ddim(
                    flow, cov_y, mu_x, cov_x, sde_type=sde_type, q_0_sampling=True
                )
            if sampling_mode == "ode":
                sampler = get_x_y_sampler_ode(
                    flow, cov_y, mu_x, cov_x, sde_type=sde_type, q_0_sampling=True
                )
            if sampling_mode == "sde":
                sampler = get_x_y_sampler_sde(
                    flow, cov_y, mu_x, cov_x, q_0_sampling=True
                )
            keys = jr.split(key_sample, n_data)
            X_ = jax.vmap(sampler)(keys, Y)

        print("mu/cov x:", mu_x, cov_x)
    else:
        X_ = jax.vmap(partial(measurement, cov_y=5. * cov_y))(keys, X) # Testing

    # Sample latents unconditionally
    if sampling_mode in ["ddim", "sde"]:
        X_test = jax.vmap(get_non_singular_sample_fn(flow))(keys)
    if sampling_mode == "ode":
        X_test = jax.vmap(get_ode_sample_fn(flow, sde_type=sde_type))(keys)

    # Plot initial samples
    plot_samples(X, X_, Y, X_test, save_dir=save_dir)

    # Expectation maximisation
    losses_k = []
    for k in range(em_iterations):
        key_k, key_sample = jr.split(jr.fold_in(key_em, k))

        # Train on sampled latents
        losses_i = []
        with trange(
            diffusion_iterations, desc="Training", colour="green"
        ) as steps:
            for i in steps:
                key_x, key_step = jr.split(jr.fold_in(key_k, i))

                x = jr.choice(key_x, X_, (n_batch,)) # Make sure always choosing x ~ p(x|y)

                L, flow, key, opt_state = make_step(
                    flow, 
                    x, 
                    key_step, 
                    opt_state, 
                    opt.update, 
                    loss_type=loss_type, 
                    time_schedule=time_schedule
                )

                if use_ema:
                    ema_flow = apply_ema(ema_flow, flow, ema_rate)

                losses_i.append(L)
                steps.set_postfix_str(f"{k=:04d} {L=:.3E}")

        # Plot losses
        losses_k += losses_i
        plot_losses(losses_k, k, diffusion_iterations, save_dir=save_dir)

        # Generate latents from q(x|y)
        if sampling_mode == "ddim":
            sampler = get_x_y_sampler_ddim(
                ema_flow if use_ema else flow, cov_y=cov_y, sde_type=sde_type, mode=mode
            )
        if sampling_mode == "ode":
            sampler = get_x_y_sampler_ode(
                ema_flow if use_ema else flow, cov_y=cov_y, sde_type=sde_type, mode=mode
            )
        if sampling_mode == "sde":
            sampler = get_x_y_sampler_sde(
                ema_flow if use_ema else flow, cov_y=cov_y, mode=mode
            )
        keys = jr.split(key_sample, n_data)
        X_ = jax.vmap(sampler)(keys, Y)

        if clip_x_y:
            X_ = clip_latents(X_, x_clip_limit) 

        # Generate latents from q(x)
        if sampling_mode in ["ddim", "sde"]:
            X_test = jax.vmap(get_non_singular_sample_fn(ema_flow if use_ema else flow))(keys)
        if sampling_mode == "ode":
            X_test = jax.vmap(get_ode_sample_fn(ema_flow if use_ema else flow, sde_type=sde_type))(keys)

        # Plot latents
        plot_samples(X, X_, Y, X_test, iteration=k + 1, save_dir=save_dir)

        if k > 1:
            create_gif(save_dir)

        # Restart optimiser on previously trained score network
        if re_init_opt_state:
            opt, opt_state = get_opt_and_state(
                flow, 
                optimiser,
                lr=lr, 
                use_lr_schedule=use_lr_schedule, 
                initial_lr=initial_lr,
                n_epochs_warmup=n_epochs_warmup,
                n_data=n_data,
                n_batch=n_batch,
                diffusion_iterations=diffusion_iterations
            )

        # ...automatically begins next k-iteration with parameters from SGM this iteration

# score_y_x = J_E_x_x_t.T @ jnp.linalg.inv(cov_y + cov_t @ J_E_x_x_t) @ (y_ - E_x_x_t) # Eq 20, 22
# return J_E_x_x_t.T @ jnp.linalg.inv(cov_y + V_x_x_t) @ (y_ - E_x_x_t) # Eq 20

    # opt                  = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-3))
    # opt                  = optax.chain(optax.clip_by_global_norm(1.), soap(1e-3)) # NOTE: schedule?


    # @typecheck
    # def sample(
    #     self, 
    #     key: PRNGKeyArray, 
    #     soln_kwargs: Optional[dict] = None
    # ) -> Array:
    #     return single_sample_fn(
    #         self, key, self.x_shape, **default(soln_kwargs, self.soln_kwargs)
    #     )

    # @typecheck
    # def sample_with_score(
    #     self, 
    #     key: PRNGKeyArray, 
    #     y_x_score: Array, 
    #     soln_kwargs: Optional[dict] = None
    # ) -> Array:
    #     # xy_score = score[p(y|x)]
    #     return single_score_sample_fn(
    #         self, 
    #         key, 
    #         self.x_shape, 
    #         y_x_score=y_x_score, 
    #         **default(soln_kwargs, self.soln_kwargs)
    #     )

    # @typecheck
    # def sample_sde(
    #     self,
    #     key: PRNGKeyArray, 
    #     *,
    #     t0: float, 
    #     t1: float, 
    #     g_scale: float = 0.1, 
    #     n_steps: int = 1000,
    #     n: int = 1, 
    #     m: int = 0
    # ) -> Array:
    #     return single_non_singular_sample_fn(
    #         self.v, 
    #         q=None, 
    #         a=None, 
    #         key=key, 
    #         x_shape=self.x_shape, 
    #         t0=default(t0, self.t0), 
    #         t1=default(t1, self.t1), 
    #         n_steps=n_steps,
    #         g_scale=g_scale,
    #         m=m,
    #         n=n
    #     )



# OLD INITIAL SMAPLERS

# @typecheck
# def get_initial_x_y_sampler_ode(
#     flow: RectifiedFlow, 
#     mu_x: XArray, 
#     cov_x: XCovariance,
#     cov_y: YCovariance
# ) -> Callable[[PRNGKeyArray, XArray], XArray]:
#     fn = lambda key, y_: sample_initial_score_ode(
#         mu_x, cov_x, cov_y, flow, key, y_
#     )
#     return fn


# @typecheck
# def get_initial_x_y_sampler_ode(
#     flow: RectifiedFlow, 
#     mu_x: XArray, 
#     cov_x: XCovariance,
#     cov_y: YCovariance,
#     *,
#     sde_type: SDEType = "non-singular"
# ) -> Callable[[PRNGKeyArray, XArray], XArray]:
#     fn = lambda key, y_: sample_initial_score_ode(
#         mu_x, cov_x, cov_y, flow, key, y_, sde_type=sde_type
#     )
#     return fn


# @typecheck
# def get_initial_x_y_sampler_ddim(
#     flow: RectifiedFlow, 
#     mu_x: XArray, 
#     cov_x: XCovariance,
#     cov_y: YCovariance,
#     mode: Literal["full", "cg"] = "full"
# ) -> Callable[[PRNGKeyArray, XArray], XArray]:
#     fn = lambda key, y_: single_x_y_ddim_sample_fn(
#         mu_x, cov_x, cov_y, flow, key, y_, q_0_sampling=True, mode=mode
#     )
#     return fn


# @typecheck
# @eqx.filter_jit
# def sample_initial_score_ode(
#     mu_x: XArray, 
#     cov_x: XCovariance, 
#     cov_y: YCovariance,
#     flow: RectifiedFlow, 
#     key: PRNGKeyArray, 
#     y_: YArray,
#     *,
#     sde_type: SDEType = "zero-ends",
# ) -> XArray:
#     # Sample from initial q_0(x|y) with PPCA prior q_0(x)

#     inv_cov_x = jnp.linalg.inv(cov_x)

#     @typecheck
#     def get_score_gaussian_y_x(
#         y_: YArray, 
#         x: XArray, # x_t
#         alpha_t: Scalar,
#         cov_t: TCovariance,
#         score: XArray, 
#         cov_y: YCovariance
#     ) -> XArray:
#         # Score of Gaussian kernel centred on data NOTE: mode="Gaussian" for init
#         E_x_x_t, J_E_x_x_t = value_and_jacfwd(get_E_x_x_t, x, alpha_t, cov_t, score) # Tweedie; mean and jacobian
#         V_x_x_t = cov_t @ J_E_x_x_t # Or heuristics; cov_t, inv(cov_t)...
#         V_y_x_t = cov_y + V_x_x_t
#         return J_E_x_x_t.T @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20

#     def reverse_ode(t, x, args):
#         # Sampling along conditional score p(x|y)
#         t = jnp.asarray(t)

#         alpha_t = flow.alpha(t)
#         cov_t = get_cov_t(flow, t)

#         # score_x = x / jnp.minimum(alpha_t, EPS) + cov_t @ inv_cov_x @ (x - mu_x) # Tweedie with score of analytic G[x|mu_x, cov_x]
#         score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_clip(alpha_t, EPS) # Tweedie with score of analytic G[x|mu_x, cov_x]

#         score_y_x = get_score_gaussian_y_x(y_, x, alpha_t, cov_t, score_x, cov_y) # This y is x_t?

#         score_x_y = score_x + score_y_x

#         return flow.reverse_ode(x, t, score=score_x_y, sde_type=sde_type) 

#     sol = dfx.diffeqsolve(
#         dfx.ODETerm(reverse_ode), 
#         **get_flow_soln_kwargs(flow, reverse=True),
#         y0=jr.normal(key, flow.x_shape) # y1
#     )
#     return sol.ys[0]




# @eqx.filter_jit
# def single_score_sample_fn(
#     flow: RectifiedFlow, 
#     key: PRNGKeyArray, 
#     y_x_score: Optional[XArray] = None,
# ) -> XArray:
#     flow = eqx.nn.inference_mode(flow, True)

#     def _flow(t, x, args):
#         score_x = velocity_to_score(flow, t, x) #(-(1. - t) * flow.v(t, x) - x) / t
#         if y_x_score is not None:
#             score_x_y = score_x + y_x_score
#         return score_x_y
    
#     y1 = jr.normal(key, flow.x_shape)

#     term = dfx.ODETerm(_flow) 
#     sol = dfx.diffeqsolve(
#         term, 
#         **get_flow_soln_kwargs(flow, reverse=True),
#         y0=y1
#     )
#     (y0,) = sol.ys

#     return y0


        # # Non-singular SDE[v(x, t)]
        # if sde_type == "non-singular":
        #     # drift = v + 0.5 * t * score * jnp.square(alpha) 
        #     # diffusion = alpha * jnp.sqrt(t)
        #     f_, g_ = t, t
        # # Zero-ends SDE[v(x, t)]
        # if sde_type == "zero-ends":
        #     # drift = v + 0.5 * t * (1. - t) * score * jnp.square(alpha) 
        #     # diffusion = alpha * jnp.sqrt(t * (1. - t))
        #     f_, g_ = t * (1. - t), t * (1. - t)
        # # Singular SDE[v(x, t)] 
        # if sde_type == "singular":
        #     # drift = v + 0.5 * t / (1. - t) * score * jnp.square(alpha) 
        #     # diffusion = alpha * jnp.sqrt(t / (1. - t))
        #     f_, g_ = t / (1. - t), t / (1. - t) 