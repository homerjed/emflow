from typing import Literal, Optional, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx

from custom_types import (
    Array, XArray, XCovariance, YArray, YCovariance, TCovariance, 
    XSampleFn, XYSampleFn, SampleType, SDEType, PRNGKeyArray,
    OperatorFn, OperatorMatrix, Scalar, PostProcessFn,
    typecheck
)
from rf import (
    RectifiedFlow, velocity_to_score, score_to_velocity, get_flow_soln_kwargs
)
from utils import exists, maybe_invert, flatten, unflatten
from posterior import (
    get_score_gaussian_y_x, 
    get_score_gaussian_x_y, 
    get_score_gaussian_x_y_cg, 
    get_score_x_y, 
    get_score_y_x_cg,
    get_score_x_y_cg
)


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
            score_x_y = get_score_gaussian_x_y(
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
    sde_type: SDEType = "zero-ends",
    postprocess_fn: PostProcessFn = None
) -> XArray:
    """ Sample ODE of non-singular SDE corresponding to Gaussian flow matching marginals """

    def ode(t: Scalar, x: XArray, args: Optional[tuple]) -> XArray:
        # Non-Singular ODE; using score of Gaussian Rectified Flow
        t = jnp.asarray(t)
        v = flow.v(t, x)
        score = velocity_to_score(flow=None, t=t, x=x, velocity=v) # NOTE: just use flow in here, no v = ... above
        return flow.reverse_ode(x, t, score, alpha=alpha, sde_type=sde_type)
    
    flow = eqx.nn.inference_mode(flow, True)

    z = jr.normal(key, flow.img_shape)

    term = dfx.ODETerm(ode) 

    sol = dfx.diffeqsolve(
        term, 
        flow.solver, 
        flow.t1, 
        flow.t0, 
        -flow.dt0, 
        z
    )
    (x,) = sol.ys

    if exists(postprocess_fn):
        x = postprocess_fn(x)

    return x


@typecheck
@eqx.filter_jit
def single_x_y_sample_fn_ode(
    flow: RectifiedFlow,
    key: PRNGKeyArray, 
    y_: YArray,
    A: Optional[OperatorMatrix],
    cov_y: YCovariance, 
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[XCovariance] = None, 
    *,
    mode: Literal["full", "cg"] = "full",
    max_steps: int = 2,
    tol: float = 1e-5,
    sde_type: SDEType = "zero-ends",
    q_0_sampling: bool = False, # Sampling initial q_0(x) or model q(x)
    postprocess_fn: PostProcessFn = None
) -> XArray:
    # Latent posterior sampling function

    cov_x = jnp.linalg.inv(inv_cov_x) # NOTE: just supply cov_x

    def reverse_ode(t: Scalar, x: XArray, args: Optional[tuple]) -> XArray:
        # Sampling along conditional score p(x|y)
        t = jnp.asarray(t)

        if q_0_sampling:
            # Implement CG method for this
            if mode == "full":
                score_x_y = get_score_gaussian_x_y(
                    y_, 
                    A, 
                    postprocess_fn(x), 
                    t, 
                    flow, 
                    cov_y=cov_y, 
                    mu_x=mu_x, 
                    inv_cov_x=inv_cov_x
                )
            if mode == "cg":
                score_x_y = get_score_gaussian_x_y_cg(
                    y_, 
                    A, 
                    postprocess_fn(x), 
                    t, 
                    flow, 
                    cov_y=cov_y, 
                    mu_x=mu_x, 
                    inv_cov_x=inv_cov_x, 
                    max_steps=max_steps,
                    tol=tol
                )
        else:
            if mode == "full":
                # NOTE: need to implement using A here...
                score_x_y = get_score_x_y(
                    y_, 
                    A, 
                    postprocess_fn(x), 
                    t, 
                    flow, 
                    cov_y, 
                    return_score_x=True # x is x_t
                ) 
            if mode == "cg":
                score_x_y = get_score_x_y_cg(
                    y_, 
                    A, 
                    postprocess_fn(x), 
                    t, 
                    flow, 
                    cov_x, 
                    cov_y, 
                    max_steps=max_steps, 
                    tol=tol
                ) 

        score_x_y = unflatten(score_x_y, flow.img_shape)

        return flow.reverse_ode(x, t, score=score_x_y, sde_type=sde_type) 

    y0 = jr.normal(key, flow.img_shape) 

    sol = dfx.diffeqsolve(
        dfx.ODETerm(reverse_ode), 
        **get_flow_soln_kwargs(flow, reverse=True),
        y0=y0
    )
    (x,) = sol.ys

    if exists(postprocess_fn):
        x = postprocess_fn(x)

    return x


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

    z = jr.normal(key_z, flow.img_shape)

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

"""
    > Posterior samplers
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
    max_steps: int = 2,
    sde_type: SDEType = "zero-ends",
    q_0_sampling: bool = False,
    postprocess_fn: PostProcessFn = None
) -> XYSampleFn:

    inv_cov_x = maybe_invert(cov_x)

    def sampler(
        key: PRNGKeyArray, 
        y_: YArray, 
        A: OperatorMatrix = None
    ) -> XArray: 
        x_y = single_x_y_sample_fn_ode(
            flow, 
            key, 
            y_, 
            A,
            cov_y, 
            mu_x,
            inv_cov_x, 
            mode=mode, 
            max_steps=max_steps,
            sde_type=sde_type,
            q_0_sampling=q_0_sampling,
            postprocess_fn=postprocess_fn # NOTE: Why does this need cov_x? Does cov_x need iterating?
        )
        return x_y

    return sampler


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
    sde_type: SDEType = "zero-ends", # Compatability (not used here)
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
def get_x_y_sampler(
    flow: RectifiedFlow, 
    sampling_mode: SampleType, 
    cov_y: YCovariance,
    mu_x: Optional[XArray] = None,
    cov_x: Optional[XCovariance] = None,
    *,
    mode: Literal["full", "cg"] = "full",
    max_steps: int = 2,
    sde_type: SDEType = "zero-ends",
    q_0_sampling: bool = False,
    postprocess_fn: PostProcessFn = None
) -> XYSampleFn:
    # Get sampler for posterior p(x|y)

    if q_0_sampling:
        assert exists(mu_x) and exists(cov_x)

    if sampling_mode == "ddim":
        sampler_fn = get_x_y_sampler_ddim
    if sampling_mode == "ode":
        sampler_fn = get_x_y_sampler_ode
    if sampling_mode == "sde":
        sampler_fn = get_x_y_sampler_sde

    sampler = sampler_fn(
        flow, 
        cov_y=cov_y, 
        mu_x=mu_x, 
        cov_x=cov_x, 
        sde_type=sde_type, 
        mode=mode,
        max_steps=max_steps,
        q_0_sampling=q_0_sampling,
        postprocess_fn=postprocess_fn
    )

    return sampler


"""
    > Latent samplers
"""


@typecheck
def get_ode_sample_fn(
    flow: RectifiedFlow,
    *,
    alpha: float = 0.1,
    sde_type: SDEType = "zero-ends",
    postprocess_fn: PostProcessFn = None
) -> XSampleFn:
    # Sampler for e.g. sampling latents from model p(x) without y 
    def sampler(key): 
        x = single_sample_fn_ode(
            flow, key=key, alpha=alpha, sde_type=sde_type, postprocess_fn=postprocess_fn
        )
        return x
    return sampler


@typecheck
def get_non_singular_sample_fn(
    flow: RectifiedFlow,
    *,
    n_steps: int = 500,
    g_scale: float = 0.1,
    n: float = 1.,
    m: float = 0.,
    sde_type: SDEType = "zero-ends" # Ignored, for compatibility
) -> XSampleFn:
    # Sampler for e.g. sampling latents from model p(x) without y 
    fn = lambda key: single_non_singular_sample_fn(
        flow, key=key, g_scale=g_scale, n_steps=n_steps, n=n, m=m
    )
    return fn


@typecheck
def get_x_sampler(
    flow: RectifiedFlow,
    sampling_mode: SampleType,
    *,
    sde_type: SDEType = "zero-ends",
    postprocess_fn: PostProcessFn = None
) -> XSampleFn:
    # Sample latents unconditionally

    if sampling_mode in ["ddim", "sde"]:
        sampler = get_non_singular_sample_fn
    if sampling_mode == "ode":
        sampler = get_ode_sample_fn

    def sampler_fn(key): 
        return sampler(
            flow, sde_type=sde_type, postprocess_fn=postprocess_fn
        )(key)

    return sampler_fn
