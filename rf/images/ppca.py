from typing import Optional, Literal
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from tqdm import trange

from custom_types import (
    XArray, XCovariance, SDEType, 
    SampleType, PRNGKeyArray, Scalar,
    Float, Array, Int, OperatorFn, 
    PostProcessFn, typecheck
)
from rf import RectifiedFlow
from sample import get_x_y_sampler
from utils import plot_losses_ppca, maybe_shard, flatten, unflatten, get_loader


def gaussian_log_prob(x: XArray, mu_x: XArray, cov_x: XCovariance) -> Scalar:
    return jax.scipy.stats.multivariate_normal.logpdf(x, mu_x, cov_x)


@typecheck
@eqx.filter_jit
def ppca(
    x: Float[Array, "n d"], 
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


def run_ppca(
    key: PRNGKeyArray, 
    flow: RectifiedFlow,
    latent_dim: int,
    Y: Float[Array, "n y"], 
    A: Int[Array, "n y x"],
    cov_y: Float[Array, "y y"],
    n_pca_iterations: int = 256,
    n_batch: int = 1000,
    *,
    mode: Literal["cg", "full"] = "full",
    max_steps: int,
    sde_type: SDEType, 
    sampling_mode: SampleType,
    postprocess_fn: PostProcessFn = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
    distributed_sharding: Optional[jax.sharding.NamedSharding] = None,
    X: Optional[Float[Array, "n x"]] = None, # Only for loss, debugging
    save_dir: str = None
) -> tuple[XArray, XCovariance, Float[Array, "n x"]]:

    flow = maybe_shard(flow, replicated_sharding)

    n_data, data_dim = Y.shape

    mu_x = jnp.zeros(latent_dim)
    cov_x = jnp.identity(latent_dim) # Start at cov_y?

    Y = maybe_shard(Y, distributed_sharding) # Already flattened
    A = maybe_shard(A, distributed_sharding) 

    # Run PPCA, 
    log_probs_x = []
    with trange(
        n_pca_iterations, desc="PPCA Training", colour="blue"
    ) as steps:
        for s in steps:
            key_pca, key_sample = jr.split(jr.fold_in(key, s))

            # Sample X|Y
            sampler = get_x_y_sampler(
                flow, 
                cov_y=cov_y, 
                mu_x=mu_x, 
                cov_x=cov_x, 
                sde_type=sde_type,
                sampling_mode=sampling_mode, 
                mode=mode, 
                max_steps=max_steps,
                postprocess_fn=postprocess_fn,
                q_0_sampling=True
            )

            keys = jr.split(key_sample, n_batch)
            X_Y = jax.vmap(sampler)(keys, Y, A) # Sampling X|Y each time, improved q_0(x)

            X_Y = maybe_shard(X_Y, distributed_sharding)

            # PPCA estimators for mu_x, cov_x
            mu_x, cov_x = ppca(X_Y, key_pca, rank=data_dim)

            # Log-likelihood of latents under q_0(x)
            if X is not None:
                log_likelihood = partial(gaussian_log_prob, mu_x=mu_x, cov_x=cov_x)
                l_x = -jnp.mean(jax.vmap(log_likelihood)(flatten(X)))
                log_probs_x.append(l_x)

            steps.set_postfix_str("L_x={:.3E}".format(l_x))

    if X is not None:
        plot_losses_ppca(log_probs_x, save_dir)

    return mu_x, cov_x, X_Y


"""
def run_ppca(
    key: PRNGKeyArray, 
    flow: RectifiedFlow,
    latent_dim: int,
    Y: Float[Array, "n y"], 
    A: Int[Array, "n y x"],
    cov_y: Float[Array, "y y"],
    n_pca_iterations: int = 256,
    n_batch: int = 1000,
    *,
    mode: Literal["cg", "full"] = "full",
    sde_type: SDEType, 
    sampling_mode: SampleType,
    postprocess_fn: PostProcessFn = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
    distributed_sharding: Optional[jax.sharding.NamedSharding] = None,
    X: Optional[Float[Array, "n x"]] = None, # Only for loss, debugging
    save_dir: str = None
) -> tuple[XArray, XCovariance, Float[Array, "n x"]]:

    flow = maybe_shard(flow, replicated_sharding)

    n_data, data_dim = Y.shape

    mu_x = jnp.zeros(latent_dim)
    cov_x = jnp.identity(latent_dim) # Start at cov_y?

    # X_Y = maybe_shard(Y, distributed_sharding)

    loader = get_loader(
        maybe_shard(Y, distributed_sharding), # Already flattened
        maybe_shard(A, distributed_sharding), # Already flattened
        key=key
    )

    # Run PPCA, 
    log_probs_x = []
    with trange(
        n_pca_iterations, desc="PPCA Training", colour="blue"
    ) as steps:
        for s, (_Y, _A) in zip(
            steps, loader.loop(n_batch)
        ):                
            key_pca, key_sample = jr.split(jr.fold_in(key, s))

            # Sample X|Y
            sampler = get_x_y_sampler(
                flow, 
                cov_y=cov_y, 
                mu_x=mu_x, 
                cov_x=cov_x, 
                sde_type=sde_type,
                sampling_mode=sampling_mode, 
                mode=mode, 
                postprocess_fn=postprocess_fn,
                q_0_sampling=True
            )

            keys = jr.split(key_sample, n_batch)
            X_Y = jax.vmap(sampler)(keys, _Y, _A)

            X_Y = maybe_shard(X_Y, distributed_sharding)

            # PPCA estimators for mu_x, cov_x
            mu_x, cov_x = ppca(X_Y, key_pca, rank=data_dim)

            # Log-likelihood of latents under q_0(x)
            if X is not None:
                log_likelihood = partial(gaussian_log_prob, mu_x=mu_x, cov_x=cov_x)
                l_x = -jnp.mean(jax.vmap(log_likelihood)(flatten(X)))
                log_probs_x.append(l_x)

            steps.set_postfix_str("L_x={:.3E}".format(l_x))

    if X is not None:
        plot_losses_ppca(log_probs_x, save_dir)

    return mu_x, cov_x, X_Y            
"""