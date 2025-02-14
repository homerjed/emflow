import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from sgm import XArray, Covariance

typecheck = jaxtyped(typechecker=typechecker)


@typecheck
@eqx.filter_jit
def ppca(
    x: Float[Array, "n 2"], key: PRNGKeyArray, rank: int = 1
) -> tuple[XArray, Covariance]:
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