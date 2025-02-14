
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from sgm import SGM, XArray

typecheck = jaxtyped(typechecker=typechecker)


@typecheck
def single_loss_fn(
    sgm: SGM,
    x: XArray, 
    t: Scalar, 
    key: PRNGKeyArray
) -> Scalar:
    noise = jr.normal(key, x.shape)
    mu_t, sigma_t = sgm.sde.p_t(x, t)
    y = mu_t + sigma_t * noise
    pred = sgm.net(t, y)
    return sgm.sde.weight(t) * jnp.mean((pred + noise / sigma_t) ** 2.)


@typecheck
def batch_loss_fn(
    sgm: SGM,
    x: Float[Array, "n 2"], 
    key: PRNGKeyArray
) -> Scalar:
    batch_size = x.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)

    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(
        tkey, 
        (batch_size,), 
        minval=sgm.soln_kwargs["t0"], 
        maxval=sgm.soln_kwargs["t1"] / batch_size
    )
    t = t + (sgm.soln_kwargs["t1"] / batch_size) * jnp.arange(batch_size)
    # t = jr.beta(key, a=3., b=3., shape=(batch_size,))

    loss_fn = jax.vmap(partial(single_loss_fn, sgm))
    return jnp.mean(loss_fn(x, t, losskey))


@eqx.filter_jit
def make_step(
    sgm: SGM,
    x: Float[Array, "n 2"], 
    key: PRNGKeyArray, 
    opt_state: optax.OptState, 
    opt_update: callable
) -> tuple[Scalar, eqx.Module, PRNGKeyArray, optax.OptState]:
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(sgm, x, key)
    updates, opt_state = opt_update(grads, opt_state, sgm)
    sgm = eqx.apply_updates(sgm, updates)
    key, _ = jr.split(key)
    return loss, sgm, key, opt_state