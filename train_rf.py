from functools import partial
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, PyTree, jaxtyped
from beartype import beartype as typechecker

from model_rf import RectifiedFlow
from utils import exists

typecheck = jaxtyped(typechecker=typechecker)

YArray = Float[Array, "2"]

YCovariance = Float[Array, "2 2"]

XArray = Float[Array, "2"]


"""
    Train
"""


def apply_ema(
    ema_model: eqx.Module, 
    model: eqx.Module, 
    ema_rate: float, 
    # policy: Optional[Policy] = None
) -> eqx.Module:
    # if exists(policy):
    #     model = policy.cast_to_param(model)
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    m_, _m = eqx.partition(model, eqx.is_inexact_array) # Current model params
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array) # Old EMA params
    e_ = jax.tree_util.tree_map(ema_fn, e_, m_) # New EMA params
    return eqx.combine(e_, _m)


def identity(t: Scalar) -> Scalar:
    return t


def cosine_time(t: Scalar) -> Scalar:
    return 1. - (1. / (jnp.tan(0.5 * jnp.pi * t) + 1.)) # t1?


def time_sampler(
    key: PRNGKeyArray, 
    n: int, 
    t0: float, 
    t1: float, 
    time_schedule: Optional[Callable[[Scalar], Scalar]] = None
) -> Scalar:
    t = jr.uniform(key, (n,), minval=t0, maxval=t1 / n)
    t = t + (t1 / n) * jnp.arange(n)
    if exists(time_schedule):
        t = time_schedule(t)
    return t


def mse(x: XArray, y: XArray) -> XArray:
    return jnp.square(jnp.subtract(x, y))


@typecheck
def batch_loss_fn(
    flow: RectifiedFlow, 
    x_0: Float[Array, "n 2"], 
    key: PRNGKeyArray, 
    *,
    sigma_min: float = 1e-4,
    loss_type: Literal["mse", "huber"] = "mse",
    time_schedule: Optional[Callable[[Scalar], Scalar]] = cosine_time
) -> tuple[Scalar, Scalar]:
    """
        Computes MSE between the conditional vector field (x1 - x0)
        and a vector field given by the neural network.
    """
    key_eps, key_t = jr.split(key)

    t = time_sampler(
        key_t, 
        x_0.shape[0], 
        t0=flow.t0, 
        t1=flow.t1, 
        time_schedule=time_schedule
    )

    x_1 = jr.normal(key_eps, x_0.shape) 

    # Flow matching

    # def single_fm_loss(flow, t, x_0, x_1):
    #     psi = (1. - t) * x_0 + (1. - (1. - sigma_min) * (1. - t)) * x_1
    #     psi = t * x_0 + (1. - (1. - sigma_min) * t) * x_1
    #     u = (1. - sigma_min) * x_1 - x_0
    #     v = flow.v(t, psi)
    #     return mse(u, v)

    # loss = jnp.mean(jax.vmap(partial(single_fm_loss, flow))(t, x_0, x_1))
    
    # Rectified
    x_1 = jr.normal(key_eps, x_0.shape) 

    x_t = jax.vmap(flow.p_t)(x_0, t, x_1) 

    v = jax.vmap(flow.v)(t, x_t) 

    if loss_type == "mse":
        loss = jnp.mean(mse(v, x_1 - x_0))
    if loss_type == "huber":
        c = 0.00054 * x_0.shape[-1]
        loss = jnp.sqrt(jnp.mean(mse(v, x_1 - x_0)) + c * c) - c

    return loss, loss


@typecheck
@eqx.filter_jit
def make_step(
    model: RectifiedFlow, 
    x: Array, 
    key: PRNGKeyArray, 
    opt_state: optax.OptState, 
    opt_update: Callable, 
    *,
    loss_type: Literal["mse", "huber"] = "mse",
    time_schedule: Callable[[Scalar], Scalar] = identity,
    grad_accumulate: bool = False,
    n_minibatches: int = 4
) -> tuple[
    Scalar, RectifiedFlow, PRNGKeyArray, optax.OptState
]:
    grad_fn = eqx.filter_value_and_grad(
        partial(
            batch_loss_fn, 
            loss_type=loss_type, 
            time_schedule=time_schedule
        ), 
        has_aux=True
    )

    if grad_accumulate:
        (loss, _), grads,  = accumulate_gradients_scan(
            model, x, key, n_minibatches=n_minibatches, grad_fn=grad_fn
        )
    else:
        (loss, _), grads = grad_fn(model, x, key)

    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    key, _ = jr.split(key)
    return loss, model, key, opt_state


def accumulate_gradients_scan(
    model: eqx.Module,
    x: Array, # Full batch, maybe tuple of arrs
    key: PRNGKeyArray,
    n_minibatches: int,
    *,
    loss_fn: Callable = None,
    grad_fn: Callable = None
) -> tuple[tuple[Scalar, Scalar], PyTree]:
    assert not (loss_fn is not None and grad_fn is not None)

    if loss_fn is not None:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # batch_size = x.inputs.shape[0]
    batch_size = x.shape[0]
    minibatch_size = batch_size // n_minibatches
    keys = jr.split(key, n_minibatches)

    def _minibatch_step(minibatch_idx):
        # Determine gradients and metrics for a single minibatch.
        _x = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
                x, 
                start_index=minibatch_idx * minibatch_size, 
                slice_size=minibatch_size, 
                axis=0
            ),
            x, # This works for tuples of batched data e.g. (x, q, a)
        )
        (_, step_metrics), step_grads = grad_fn(model, _x, keys[minibatch_idx])
        return step_grads, step_metrics

    def _scan_step(carry, minibatch_idx):
        # Scan step function for looping over minibatches.
        step_grads, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree.map(jnp.add, carry, (step_grads, step_metrics))
        return carry, None

    # Determine initial shapes for gradients and metrics.
    grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)

    # Loop over minibatches to determine gradients and metrics.
    (grads, metrics), _ = jax.lax.scan(
        _scan_step, 
        init=(grads, metrics), 
        xs=jnp.arange(n_minibatches), 
        length=n_minibatches
    )

    # Average gradients over minibatches.
    grads = jax.tree.map(lambda g: g / n_minibatches, grads)
    metrics = jax.tree.map(lambda m: m / n_minibatches, metrics)
    return metrics, grads