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
from ml_collections import ConfigDict

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


def unflatten(x: Array, img_shape: tuple[int]) -> Array:
    c, h, w = img_shape
    return rearrange(x, "... (h w c) -> ... c h w", h=h, w=w, c=c)


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


def plot_samples(X, X_Y, Y, X_, iteration=0, save_dir="imgs/"):

    X = rearrange(X[:16], "(p q) c h w -> (p h) (q w) c", p=4, q=4)
    X_Y = rearrange(X_Y[:16], "(p q) (c h w) -> (p h) (q w) c", p=4, q=4, h=28, w=28)
    Y = rearrange(Y[:16], "(p q) (c h w) -> (p h) (q w) c", p=4, q=4, h=28, w=28)
    X_ = rearrange(X_[:16], "(p q) (c h w) -> (p h) (q w) c", p=4, q=4, h=28, w=28)

    fig, axs = plt.subplots(2, 2, figsize=(9., 9.), dpi=200)
    for ax, arr, title in zip(
        axs.ravel(), (X, Y, X_Y, X_), ("X", "Y", "X|Y", "X'")
    ):
        ax.set_title(title)
        ax.imshow(arr, cmap="gray_r")
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
        self, X: Array, A: Optional[Array] = None, *, key: PRNGKeyArray 
    ):
        self.X = X 
        self.A = A 
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
                yield (
                    (self.X[batch_perm], self.A[batch_perm]) 
                    if exists(self.A) 
                    else self.X[batch_perm]
                )
                start = end
                end = start + batch_size


def get_loader(
    X: Array, A: Optional[Array] = None, *, key: PRNGKeyArray
) -> InMemoryDataLoader:
    # Train loader only for now
    return InMemoryDataLoader(X, A, key=key)


def mnist(
    key: PRNGKeyArray, 
    img_size: int = 28,
    n_samples: int = 1000, # Testing
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
    imgs_t = 2. * (imgs - imgs.min()) / (imgs.max() - imgs.min()) - 1.
    train_dataloader = InMemoryDataLoader(imgs_t, key=key_train)

    imgs = dataset["test"]["image"]
    imgs = jnp.asarray(imgs)
    imgs = imgs[:n_samples, jnp.newaxis, ...]
    imgs = _reshape_fn(imgs)
    imgs_v = 2. * (imgs - imgs.min()) / (imgs.max() - imgs.min()) - 1.
    valid_dataloader = InMemoryDataLoader(imgs_v, key=key_valid)

    print("DATA:", imgs.shape, imgs_t.dtype, imgs_t.min(), imgs_t.max())

    del imgs

    if return_arrays:
        return imgs_t
    else:
        return train_dataloader, valid_dataloader


def get_data(key: PRNGKeyArray, config: ConfigDict) -> Float[Array, "n _ _ _"]:

    if config.data.dataset == "mnist":
        key_X, key_A = jr.split(key)

        X = mnist(
            key_X, 
            img_size=config.data.img_size, 
            n_samples=config.data.n_data, 
            return_arrays=True
        )

        n_data, *img_dim = X.shape 
        img_dim = math.prod(img_dim)

        def binary_mask(key):
            zeros = jnp.zeros((int((1. - config.data.mask_fraction) * img_dim),))
            ones = jnp.ones((int(config.data.mask_fraction * img_dim),))
            ones_and_zeros = jnp.concatenate([zeros, ones])
            return jnp.diag(jr.permutation(key, ones_and_zeros))

        keys = jr.split(key_A, n_data)
        A = jax.vmap(binary_mask)(keys).astype(jnp.int32)

    return X, A


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