import time
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
import optax
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange

from sgm import SGM, SDE, ScoreNet, VP, XArray, beta_integral
from train import make_step
from sample import single_sample_fn
from utils import ppca

typecheck = jaxtyped(typechecker=typechecker)

Covariance = Float[Array, "2 2"]


def get_data(key: PRNGKeyArray, n: int) -> Float[Array, "n 2"]:
    X, _ = make_moons(
        n, 
        noise=0.04, 
        random_state=int(jnp.sum(jr.key_data(key)))
    )
    s = StandardScaler()
    X = s.fit_transform(X)
    return jnp.asarray(X)


@typecheck
def get_score_y_x_cg(
    y_: XArray, 
    x: XArray, 
    t: Scalar, 
    sgm: SGM,
    cov_x: Covariance,
    cov_y: Covariance,
    *,
    max_iter: int = 5
) -> XArray:

    cov_t = jnp.identity(x.size) * jnp.square(sgm.sde.p_t_sigma_t(t)) 

    x, vjp = jax.vjp(lambda x_t: sgm.net(t, x_t), x) 
    y = x # If no A, x is E[x|x_t]

    # Get linear operator Mv = b to solve for v given b, choosing heuristic for V[x|x_t]
    if cov_x is None:
        cov_y_xt = lambda v: cov_y @ v + cov_t * vjp(v) # Is this right?
    else:
        cov_x_xt = cov_t + (-(cov_t ** 2.)) * jnp.linalg.inv(cov_x + cov_t)
        cov_y_xt = lambda v: cov_y @ v + cov_x_xt @ v

    b = y_ - y # y_ is data
    v, _ = jax.scipy.sparse.linalg.cg(
        A=cov_y_xt, # This is a linear operator in lineax?
        b=b,
        tol=1e-5,
        maxiter=max_iter
    )

    (score,) = vjp(v) 

    return x + cov_t @ score 


@typecheck
def value_and_jacfwd(
    f: Callable, x: XArray, cov: Covariance, score: XArray
) -> tuple[XArray, Covariance]:
    pushfwd = partial(jax.jvp, lambda x: f(x, cov, score), (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


@typecheck
def get_E_x_x_t(x_t: XArray, cov_t: Covariance, score: XArray) -> XArray: 
    return x_t + cov_t @ score # Tweedie


@typecheck
def get_score_y_x(
    y_: XArray, 
    x: XArray, # x_t
    t: Scalar, 
    sgm: SGM,
    cov_y: Covariance
) -> XArray:
    # Score of Gaussian linear data-likelihood G[y|x, cov_y]
    cov_t = jnp.eye(x.size) * jnp.square(sgm.sde.p_t_sigma_t(t))
    E_x_x_t, dE_x_x_t = value_and_jacfwd(get_E_x_x_t, x, cov_t, sgm.net(t, x))
    return dE_x_x_t.T @ jnp.linalg.inv(cov_y + cov_t @ dE_x_x_t) @ (y_ - E_x_x_t) # Eq 20, 22


@typecheck
@eqx.filter_jit
def single_x_y_sample_fn(
    sgm: SGM,
    cov_x: Covariance, 
    cov_y: Covariance, 
    key: PRNGKeyArray, 
    y_: XArray,
    *,
    mode: Literal["full", "cg"] = "full"
) -> XArray:
    # Latent posterior sampling function

    def reverse_ode(t, y, args):
        # Sampling along conditional score p(x|y)
        t = jnp.asarray(t)
        score_x = sgm.net(t, y)
        if mode == "full":
            score_y_x = get_score_y_x(y_, y, t, sgm, cov_y) # This y is x_t
        if mode == "cg":
            score_y_x = get_score_y_x_cg(y_, y, t, sgm, cov_x, cov_y) 
        score_x_y = score_x + score_y_x
        return sde.reverse_ode(score_x_y, y, t) 

    sol = dfx.diffeqsolve(
        dfx.ODETerm(reverse_ode), 
        sgm.soln_kwargs["solver"],
        sgm.soln_kwargs["t1"],
        sgm.soln_kwargs["t0"],
        -sgm.soln_kwargs["dt"],
        y0=jr.normal(key, sgm.x_shape) 
    )
    return sol.ys[0]


@typecheck
@eqx.filter_jit
def sample_initial_score(
    mu_x: XArray, 
    cov_x: Covariance, 
    sde: SDE, 
    key: PRNGKeyArray, 
    y_: XArray
) -> XArray:

    cov_inv_x = jnp.linalg.inv(cov_x)

    @typecheck
    def get_score_gaussian_y_x(
        y_: XArray, 
        y: XArray, # x_t
        cov_t: Covariance,
        score: XArray, 
        cov_y: Covariance
    ) -> XArray:
        # Score of Gaussian kernel centred on data 
        E_x_x_t, dE_x_x_t = value_and_jacfwd(get_E_x_x_t, y, cov_t, score) # Tweedie; mean and jacobian
        V_x_x_t = cov_t @ dE_x_x_t # Or heuristics; cov_t, inv(cov_t)...
        return dE_x_x_t.T @ jnp.linalg.inv(cov_y + V_x_x_t) @ (y_ - E_x_x_t) # Eq 20

    def reverse_ode(t, y, args):
        # Sampling along conditional score p(x|y)
        t = jnp.asarray(t)

        cov_t = jnp.eye(y.size) * jnp.square(sde.p_t_sigma_t(t))

        score_x = y + cov_t @ cov_inv_x @ (y - mu_x) # Tweedie with score of analytic G[x|mu_x, cov_x]
        score_y_x = get_score_gaussian_y_x(y_, y, cov_t, score_x, cov_y) # This y is x_t?

        score_x_y = score_x + score_y_x
        return sde.reverse_ode(score_x_y, y, t) 
    
    sol = dfx.diffeqsolve(
        dfx.ODETerm(reverse_ode), 
        sgm.soln_kwargs["solver"],
        sgm.soln_kwargs["t1"],
        sgm.soln_kwargs["t0"],
        -sgm.soln_kwargs["dt"],
        y0=jr.normal(key, sgm.x_shape) # y1
    )
    return sol.ys[0]


def plot_losses(losses_k):
    plt.figure()
    plt.loglog(losses_k)
    plt.savefig("imgs/L.png")
    plt.close()


def plot_samples(X, X_, Y, n_plot=8000, iteration=0):
    plt.figure(figsize=(4., 4.), dpi=200)
    plt.scatter(*X[:n_plot].T, s=0.05, marker=".", color="k", label=r"$x\sim p(x)$")
    plt.scatter(*X_[:n_plot].T, s=0.05, marker=".", color="royalblue", label=r"$x\sim p_{\theta}(x)$")
    plt.scatter(*Y[:n_plot].T, s=0.05, marker=".", color="goldenrod", label=r"$y\sim p(y|x)$")
    plt.legend(frameon=False)
    plt.savefig("imgs/samples_{:04d}.png".format(iteration))
    plt.close()


@typecheck
def get_x_y_sampler(
    sgm: SGM, 
    cov_x: Covariance, 
    cov_y: Covariance, 
    mode: Literal["full", "cg"] = "full"
) -> Callable[[PRNGKeyArray, XArray], XArray]:
    fn = lambda key, y_: single_x_y_sample_fn(
        sgm, cov_x, cov_y, key, y_, mode=mode
    )
    return fn


@typecheck
def get_initial_x_y_sampler(
    sgm: SGM, 
    mu_x: XArray, 
    cov_x: Covariance
) -> Callable[[PRNGKeyArray, XArray], XArray]:
    fn = lambda key, y_: sample_initial_score(
        mu_x, cov_x, sgm.sde, key, y_
    )
    return fn


def measurement(key: PRNGKeyArray, x: XArray, cov_y: Covariance) -> XArray: 
    return jr.multivariate_normal(key, x, cov_y) # G[y|x, cov_y]


if __name__ == "__main__":
    key = jr.key(int(time.time()))

    # Train
    em_iterations        = 64
    diffusion_iterations = 5_000
    n_batch              = 5_000
    opt                  = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-3))
    ppca_pretrain        = True
    n_pca_iterations     = 10
    clip_x_y             = True
    re_init_opt_state    = True
    n_plot               = 8000
    mode                 = "full"

    # Model
    width_size           = 256 #128
    depth                = 2 #5
    activation           = jax.nn.silu # gelu
    soln_kwargs          = dict(t0=0., dt=0.02, t1=1., solver=dfx.Heun())

    # Data
    data_dim             = 2
    n_data               = 100_000
    sigma_y              = 0.02
    cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)

    key_data, key_measurement, key_net, key_ppca, key_em = jr.split(key, 5)

    # SDE and diffusion model
    sde = VP(beta_integral)

    net = ScoreNet(
        soln_kwargs["t1"], 
        in_size=data_dim, 
        out_size=data_dim, 
        time_embedding_dim=8,
        width_size=width_size, 
        depth=depth, 
        activation=activation, 
        key=key_net
    )

    sgm = SGM(net, sde, x_shape=(data_dim,), soln_kwargs=soln_kwargs)

    opt_state = opt.init(eqx.filter(sgm, eqx.is_array))

    # Latents
    X = get_data(key_data, n_data)

    # Generate y ~ G[y|x, cov_y]
    keys = jr.split(key_measurement, n_data)
    Y = jax.vmap(partial(measurement, cov_y=cov_y))(keys, X) 

    # PPCA pre-training for q_0(x|mu_x, cov_x)
    if ppca_pretrain:
        # Initial parameters
        mu_x = jnp.zeros(data_dim)
        cov_x = jnp.identity(data_dim) # Start at cov_y?

        X_ = Y
        for s in trange(n_pca_iterations, desc="PPCA Training", colour="blue"):
            key_pca, key_sample = jr.split(jr.fold_in(key, s))

            mu_x, cov_x = ppca(X_, key_pca, rank=data_dim)

            keys = jr.split(key_sample, n_data)
            X_ = jax.vmap(get_initial_x_y_sampler(sgm, mu_x, cov_x))(keys, Y) 

        print("mu/cov x:", mu_x, cov_x)
    else:
        X_ = jax.vmap(partial(measurement, cov_y=2. * cov_y))(keys, X) # Testing

    # Plot initial samples
    plot_samples(X, X_, Y)

    # Expectation maximisation
    losses_k = []
    for k in range(em_iterations):
        key_k, key_sample = jr.split(jr.fold_in(key_em, k))

        # Train on sampled latents
        losses_i = []
        with trange(diffusion_iterations, desc="Training", colour="green") as steps:
            for i in steps:
                key_x, key_step = jr.split(jr.fold_in(key_k, i))

                x = jr.choice(key_x, X_, (n_batch,)) # Make sure always choosing x ~ p(x|y)

                L, sgm, key, opt_state = make_step(
                    sgm, x, key_step, opt_state, opt.update
                )

                losses_i.append(L)
                steps.set_postfix_str(f"\r {k=:04d} {L=:.3E}")

        # Restart optimiser on previously trained score network
        if re_init_opt_state:
            opt_state = opt.init(eqx.filter(sgm, eqx.is_array))

        # Generate latents from q(x|y)
        sample_fn = get_x_y_sampler(sgm, cov_x, cov_y, mode=mode)

        keys = jr.split(key_sample, n_data)
        X_ = jax.vmap(sample_fn)(keys, Y)

        if clip_x_y:
            X_ = X_[jnp.all(jnp.logical_and(-4. < X_, X_ < 4.), axis=-1)] # Training set gets smaller...

        # Plot latents and losses
        plot_samples(X, X_, Y, iteration=k + 1)

        losses_k += losses_i
        plot_losses(losses_k)

        # ...automatically begins next k-iteration with parameters from SGM this iteration