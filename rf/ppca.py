from typing import Optional
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from tqdm import trange

from custom_types import (
    XArray, XCovariance, SDEType, 
    SampleType, PRNGKeyArray, 
    Float, Array, typecheck
)
from rf import RectifiedFlow
from sample import get_x_y_sampler
from utils import plot_losses_ppca


def gaussian_log_prob(x, mu_x, cov_x):
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
    Y: Float[Array, "n d"], 
    cov_y: Float[Array, "d d"],
    sde_type: SDEType, 
    sampling_mode: SampleType,
    n_pca_iterations: int = 256,
    X: Optional[Float[Array, "n d"]] = None, # Only for loss, debugging
) -> tuple[XArray, XCovariance, Float[Array, "n d"]]:

    n_data, data_dim = Y.shape

    mu_x = jnp.zeros(data_dim)
    cov_x = jnp.identity(data_dim) # Start at cov_y?

    X_Y = Y

    log_probs_x = []
    with trange(
        n_pca_iterations, desc="PPCA Training", colour="blue"
    ) as steps:
        for s in steps:                
            key_pca, key_sample = jr.split(jr.fold_in(key, s))

            sampler = get_x_y_sampler(
                flow, 
                cov_y=cov_y, 
                mu_x=mu_x, 
                cov_x=cov_x, 
                sde_type=sde_type,
                sampling_mode=sampling_mode, 
                q_0_sampling=True
            )
            keys = jr.split(key_sample, n_data)
            X_Y = jax.vmap(sampler)(keys, Y)

            mu_x, cov_x = ppca(X_Y, key_pca, rank=data_dim)

            if X is not None:
                log_likelihood = partial(gaussian_log_prob, mu_x=mu_x, cov_x=cov_x)
                l_x = -jnp.mean(jax.vmap(log_likelihood)(X))
                log_probs_x.append(l_x)

            steps.set_postfix_str("L_x={:.3E}".format(l_x))

    print("mu/cov x: {} \n {}".format(mu_x, cov_x))

    if X is not None:
        plot_losses_ppca(log_probs_x)
    return mu_x, cov_x, X_Y
