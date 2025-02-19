import os
from typing import Optional, Union
from shutil import rmtree
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, Array, Float, jaxtyped
from beartype import beartype as typechecker
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
# import tensorflow_probability.substrates.jax.distributions as tfd

XArray = Float[Array, "2"]

XCovariance = Float[Array, "2 2"]

typecheck = jaxtyped(typechecker=typechecker)

EPS = 1e-10 


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def maybe_clip(v, eps=EPS):
    return jnp.maximum(v, eps) if exists(eps) else v


def maybe_invert(cov_x):
    if exists(cov_x): 
        inv_cov_x = jnp.linalg.inv(cov_x)
    else:
        inv_cov_x = None
    return inv_cov_x


def gaussian_log_prob(x, mu_x, cov_x):
    return jax.scipy.stats.multivariate_normal.logpdf(x, mu_x, cov_x)


@typecheck
@eqx.filter_jit
def ppca(
    x: Float[Array, "n 2"], 
    key: PRNGKeyArray, 
    rank: int 
) -> tuple[XArray, XCovariance]:
    # Probabilistic PCA

    samples, features = x.shape

    mu_x = jnp.mean(x, axis=0)
    x = x - mu_x

    if samples < features:
        C = x @ x.T / samples
    else:
        C = x.T @ x / samples # Sample covariance

    if rank < len(C) // 5:
        Q = jr.normal(key, (len(C), rank))
        L, Q, _ = jax.experimental.sparse.linalg.lobpcg_standard(C, Q)
    else:
        L, Q = jnp.linalg.eigh(C)
        L, Q = L[-rank:], Q[:, -rank:]

    if samples < features:
        Q = x.T @ Q
        Q = Q / jnp.linalg.norm(Q, axis=0)

    if rank < features:
        D = (jnp.trace(C) - jnp.sum(L)) / (features - rank)
    else:
        D = jnp.asarray(1e-6)

    # U = Q * jnp.sqrt(jnp.maximum(L - D, 0.0))

    # cov_x = jnp.cov(D, rowvar=False)
    # cov_x = DPLR(D * jnp.ones(features), U, U.T)

    return mu_x, C #jnp.eye(features) * D #C # cov_x


"""
    Plots & utils
"""


def clear_and_get_results_dir(save_dir: str) -> str:
    # Image save directories
    rmtree(save_dir, ignore_errors=True) # Clear old ones
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir 


def plot_losses(losses_k, iteration, diffusion_iterations, save_dir="imgs/"):
    losses_k = jnp.asarray(losses_k)
    # losses_k = losses_k[jnp.abs(losses_k - jnp.mean(losses_k)) < 2. * jnp.std(losses_k)]
    plt.figure()
    for i in range(1, iteration + 1):
        plt.axvline(i * diffusion_iterations, linestyle=":", color="gray")
    plt.loglog(losses_k)
    plt.ylabel(r"$\mathcal{L}$")
    plt.xlabel(r"$n_{steps}$")
    plt.savefig(os.path.join(save_dir, "L.png"))
    plt.close()


def plot_losses_ppca(losses, save_dir="imgs/"):
    plt.figure()
    plt.loglog(losses)
    plt.ylabel(r"$\mathcal{L}$")
    plt.xlabel(r"$n_{steps}$")
    plt.savefig(os.path.join(save_dir, "L_ppca.png"))
    plt.close()


def plot_samples(X, X_Y, Y, X_, n_plot=8000, iteration=0, save_dir="imgs/"):
    plot_kwargs = dict(s=0.08, marker=".")
    fig, axs = plt.subplots(2, 1, figsize=(4., 9.), dpi=200)
    ax = axs[0]
    ax.set_title("EM: iteration {}".format(iteration))
    ax.scatter(*X[:n_plot].T, color="k", label=r"$x\sim p(x)$", **plot_kwargs)
    ax.scatter(*X_Y[:n_plot].T, color="b", label=r"$x\sim p_{\theta}(x|y)$", **plot_kwargs) 
    ax.scatter(*Y[:n_plot].T, color="r", label=r"$y\sim p(y|x)$", **plot_kwargs) 
    legend = ax.legend(frameon=False, loc="upper right")
    legend.get_texts()[0].set_color("k") 
    legend.get_texts()[1].set_color("b") 
    legend.get_texts()[2].set_color("r") 
    ax = axs[1]
    ax.scatter(*X[:n_plot].T, color="k", label=r"$x\sim p(x)$", **plot_kwargs) 
    ax.scatter(*X_[:n_plot].T, color="b", label=r"$x\sim p_{\theta}(x)$", **plot_kwargs) 
    legend = ax.legend(frameon=False, loc="upper right")
    legend.get_texts()[0].set_color("k") 
    legend.get_texts()[1].set_color("b") 
    for ax in axs:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis("off")
    plt.savefig(
        os.path.join(save_dir, "samples_{:04d}.png".format(iteration)), 
        bbox_inches="tight"
    )
    plt.close()


def create_gif(image_folder):
    # Animate training iterations 
    images = sorted(
        [
            os.path.join(image_folder, img) 
            for img in os.listdir(image_folder) 
            if img.endswith(("png")) and "L" not in img
        ]
    )
    frames = [Image.open(img) for img in images]
    frames[0].save(
        os.path.join(image_folder, "training.gif"), 
        save_all=True, 
        append_images=frames[1:], 
        duration=500, 
        loop=0
    )


def get_opt_and_state(
    flow: eqx.Module, 
    optimiser: Union[
        optax.GradientTransformation, 
        optax.GradientTransformationExtraArgs
    ] = optax.adamw, 
    lr: float = 1e-3, 
    use_lr_schedule: bool = False, 
    initial_lr: Optional[float] = 1e-6, 
    n_data: Optional[int] = None,
    n_batch: Optional[int] = None,
    n_epochs_warmup: Optional[int] = None,
    diffusion_iterations: Optional[int] = None
) -> tuple[
    optax.GradientTransformationExtraArgs, optax.OptState
]:
    if use_lr_schedule:
        n_steps_per_epoch = int(n_data / n_batch)

        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=initial_lr, 
            peak_value=lr, 
            warmup_steps=n_epochs_warmup * n_steps_per_epoch,
            decay_steps=diffusion_iterations * n_steps_per_epoch, 
            end_value=lr
        )
    else:
        scheduler = None

    opt = optimiser(scheduler if exists(scheduler) else lr)
    opt = optax.chain(optax.clip_by_global_norm(1.), opt) 

    opt_state = opt.init(eqx.filter(flow, eqx.is_array))

    return opt, opt_state


def clip_latents(X, x_clip_limit):
    if exists(x_clip_limit):
        X = X[jnp.all(jnp.logical_and(-x_clip_limit < X, X < x_clip_limit), axis=-1)] 
    return X


"""
    Data
"""


def get_data(key: PRNGKeyArray, n: int) -> Float[Array, "n d"]:
    seed = int(jnp.sum(jr.key_data(key)))
    X, _ = make_moons(n, noise=0.04, random_state=seed)

    # N = 8 # Mixture components
    # alphas = jnp.linspace(0, 2. * jnp.pi * (1. - 1. / N), N)
    # xs = jnp.cos(alphas)
    # ys = jnp.sin(alphas)
    # means = jnp.stack([xs, ys], axis=1)

    # gaussian_mixture = tfd.Mixture(
    #     cat=tfd.Categorical(
    #         probs=jnp.ones((means.shape[0])) / means.shape[0]
    #     ),
    #     components=[
    #         tfd.MultivariateNormalDiag(
    #             loc=mu, scale_diag=jnp.ones_like(mu) * 0.1
    #         )
    #         for mu in means
    #     ]
    # )
    # X = gaussian_mixture.sample((n,), seed=key)

    s = StandardScaler() # Need to keep doing this for each EM's x|y?
    X = s.fit_transform(X)

    return jnp.asarray(X)


def measurement(
    key: PRNGKeyArray, 
    x: Float[Array, "d"], 
    cov_y: Float[Array, "d d"]
) -> Float[Array, "d"]: 
    # Sample from G[y|x, cov_y]
    return jr.multivariate_normal(key, x, cov_y) 