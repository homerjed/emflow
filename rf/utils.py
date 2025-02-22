import os
import math
import abc
from typing import Optional, Union, Generator
from shutil import rmtree

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
import tensorflow_probability.substrates.jax.distributions as tfd
from datasets import load_dataset

from custom_types import PRNGKeyArray, Array, Float, PyTree, XCovariance, Datasets, typecheck

EPS = None


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def maybe_clip(v, eps=EPS):
    return jnp.maximum(v, eps) if exists(eps) else v


def maybe_invert(cov_x: XCovariance) -> XCovariance:
    if exists(cov_x): 
        inv_cov_x = jnp.linalg.inv(cov_x)
    else:
        inv_cov_x = None
    return inv_cov_x


def flatten(x: Array) -> Array:
    return rearrange(x, "... c h w -> ... (h w c)")


def unflatten(x: Array, height: int, width: int) -> Array:
    return rearrange(x, "... (h, w, c) -> ... h w c", h=height, w=width)


"""
    Sharding
"""


def get_shardings():
    mesh = jax.sharding.Mesh(jax.devices(), "x")
    replicated = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    distributed = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x")
    )
    return replicated, distributed


def maybe_shard(
    pytree: PyTree, 
    sharding: Optional[jax.sharding.NamedSharding]
) -> PyTree:
    return eqx.filter_shard(pytree, sharding) if exists(sharding) else pytree


"""
    Plots & utils
"""


def clear_and_get_results_dir(save_dir: str, clear_previous: bool = True) -> str:
    # Image save directories
    if clear_previous:
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


def plot_samples(X, X_Y, Y, X_, dataset, n_plot=8000, iteration=0, save_dir="imgs/"):
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
        # X = rearrange(X, "(p q) c h w -> (p h) (q w) c")
        # X_Y = rearrange(X_Y, "(p q) c h w -> (p h) (q w) c")
        # Y = rearrange(Y, "(p q) c h w -> (p h) (q w) c")
        # X_ = rearrange(X_, "(p q) c h w -> (p h) (q w) c")
        # fig, axs = plt.subplots(2, 2, figsize=(9., 9.), dpi=200)
        # for ax, arr in zip(axs.ravel(), (X, X_Y, Y, X_)):
        #     ax.imshow(arr, cmap="gray_r")
        #     ax.axis("off")
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
            if (
                img.startswith(("samples")) 
                and img.endswith(("png")) 
                and "L" not in img
            )
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
        # n_steps_per_epoch = int(n_data / n_batch)
        # scheduler = optax.warmup_cosine_decay_schedule(
        #     init_value=initial_lr, 
        #     peak_value=lr, 
        #     warmup_steps=n_epochs_warmup * n_steps_per_epoch,
        #     decay_steps=diffusion_iterations * n_steps_per_epoch, 
        #     end_value=lr
        # )
        scheduler = optax.linear_schedule(
            init_value=initial_lr, 
            end_value=lr,
            transition_steps=diffusion_iterations
        )
    else:
        scheduler = None

    opt = optimiser(scheduler if exists(scheduler) else lr)
    opt = optax.chain(optax.clip_by_global_norm(1.), opt) 

    opt_state = opt.init(eqx.filter(flow, eqx.is_array))

    return opt, opt_state


def clip_latents(X: Array, x_clip_limit: float) -> Array:
    if exists(x_clip_limit):
        X = X[jnp.all(jnp.logical_and(-x_clip_limit < X, X < x_clip_limit), axis=-1)] 
    return X


"""
    Data
"""


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, X: Array, *, key: PRNGKeyArray 
    ):
        self.X = X 
        self.key = key

    def loop(
        self, 
        batch_size: int, 
        *, 
        key: Optional[PRNGKeyArray] = None
    ) -> Generator[Array, None, None]:
        dataset_size = self.X.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = key if exists(key) else self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield self.X[batch_perm]
                start = end
                end = start + batch_size


def get_loader(X, key):
    # Train loader only for now
    return InMemoryDataLoader(X, key=key)


def mnist(
    key: PRNGKeyArray, 
    img_size: int = 28,
    n_samples: int = 100, # Testing
    return_arrays: bool = True
) -> tuple[InMemoryDataLoader, InMemoryDataLoader]:

    key_train, key_valid = jr.split(key)

    def _reshape_fn(imgs):
        if img_size != 28:
            imgs = jax.image.resize(
                imgs, (imgs.shape[0], 1, img_size, img_size), method="bilinear"
            )
        return imgs

    dataset = load_dataset("ylecun/mnist")
    dataset = dataset.with_format("jax")

    imgs = dataset["train"]["image"]
    imgs = jnp.asarray(imgs)
    imgs = imgs[:n_samples, jnp.newaxis, ...]
    imgs = _reshape_fn(imgs)
    imgs_t = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    train_dataloader = InMemoryDataLoader(imgs_t, key=key_train)

    imgs = dataset["test"]["image"]
    imgs = jnp.asarray(imgs)
    imgs = imgs[:n_samples, jnp.newaxis, ...]
    imgs = _reshape_fn(imgs)
    imgs_v = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    valid_dataloader = InMemoryDataLoader(imgs_v, key=key_valid)

    print("DATA:", imgs.shape, imgs.dtype, imgs.min(), imgs.max())

    del imgs

    if return_arrays:
        return imgs_t
    else:
        return train_dataloader, valid_dataloader


def get_data(key: PRNGKeyArray, n: int, dataset: Datasets) -> Float[Array, "n d"]:

    if dataset == "blob":
        X = jr.multivariate_normal(
            key, mean=jnp.ones((2,)), cov=jnp.identity(2) * 0.1, shape=(n,)
        )
        A = None

    if dataset == "double-blob":
        key_0, key_1 = jr.split(key)

        X_0 = jr.multivariate_normal(
            key_0, mean=jnp.ones((2,)) * 1., cov=jnp.identity(2) * 0.05, shape=(n // 2,)
        )

        X_1 = jr.multivariate_normal(
            key_1, mean=jnp.ones((2,)) * -1.0, cov=jnp.identity(2) * 0.05, shape=(n // 2,)
        )

        X = jnp.concatenate([X_0, X_1])
        A = None

    if dataset == "moons":
        seed = int(jnp.sum(jr.key_data(key)))
        X, _ = make_moons(n, noise=0.04, random_state=seed)
        A = None

    if dataset == "gmm":
        N = 8 # Mixture components

        alphas = jnp.linspace(0., 2. * jnp.pi * (1. - 1. / N), N)
        xs = jnp.cos(alphas)
        ys = jnp.sin(alphas)
        means = jnp.stack([xs, ys], axis=1)

        gaussian_mixture = tfd.Mixture(
            cat=tfd.Categorical(
                probs=jnp.ones((means.shape[0])) / means.shape[0]
            ),
            components=[
                tfd.MultivariateNormalDiag(
                    loc=mu, scale_diag=jnp.ones_like(mu) * 0.1
                )
                for mu in means
            ]
        )
        X = gaussian_mixture.sample((n,), seed=key)
        A = None

    if dataset == "spiral":
        X, _ = make_swiss_roll(n, noise=0.01)  # X is (n_samples, 3)
        X = 2. * (X - X.min()) / (X.max() - X.min()) - 1.  # Normalize to [0, 1]
        A = None

        # fig = plt.figure(figsize=(12, 5))
        # ax = fig.add_subplot(121, projection='3d')
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0], cmap='Spectral')
        # ax.set_title("Original 3D Swiss Roll")

        # noise = np.random.normal(scale=0.02, size=(n_samples, 2))  # Gaussian noise
        # Y = np.einsum("nij, nj -> ni", A, X) + noise  

        # ax = fig.add_subplot(122)
        # ax.scatter(Y[:, 0], Y[:, 1], c=X[:, 0], cmap='Spectral')
        # ax.set_title("Projected & Noisy 2D Data")
        # plt.show()

    if dataset == "mnist":
        X = mnist(key, img_size=28, return_arrays=True)
        # A = jr.choice(
        #     key, 
        #     jnp.array([0, 1]), 
        #     p=jnp.array([0.75, 1. - 0.75]),
        #     shape=(X.shape[0], math.prod(X.shape[1:]))
        # )
        img_dim = math.prod(X.shape[1:])
        A = jr.permutation(
            key, 
            jnp.stack([jnp.array([0.] * int(0.75 * img_dim) + [1.] * int(0.25 * img_dim))] * X.shape[0]),
            independent=True
        )
        A = jax.vmap(jnp.diag)(A).astype(jnp.int32)

    # NOTE: shuffle so a representative set are used / plotted in all cases
    X = jr.permutation(key, X)

    # if dataset != "blob":
    #     s = StandardScaler() # Need to keep doing this for each EM's x|y?
    #     X = s.fit_transform(X)

    return jnp.asarray(X), A


def get_A(key: PRNGKeyArray, latent_dim: int = 3, observed_dim: int = 2) -> Float[Array, "y x"]:
    # Generate corruption matrix A, data_dim is latent variable size
    # one_and_zeros = jnp.array([1.] * observed_dim + [0.] * (latent_dim - observed_dim)) # Assuming n 
    # A = jnp.diag(jr.permutation(key, one_and_zeros,))
    idx = jr.choice(key, latent_dim, shape=(observed_dim,), replace=False)  # Sample m indices
    A = jnp.zeros((observed_dim, latent_dim)).at[jnp.arange(observed_dim), idx].set(1)
    return A.astype(jnp.int32)


def measurement(
    key: PRNGKeyArray, 
    x: Float[Array, "x"], 
    *,
    A: Optional[Float[Array, "y x"]], 
    cov_y: Float[Array, "y y"]
) -> Float[Array, "d"]: 
    # Sample from G[y|x, cov_y]
    mu_y = A @ x if exists(A) else x
    return jr.multivariate_normal(key, mu_y, cov_y) 