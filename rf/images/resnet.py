
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar

from custom_types import XArray, SDEType, typecheck
from utils import exists, maybe_clip


class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        *, 
        key: PRNGKeyArray 
    ):
        lim = jnp.sqrt(1. / (in_size + 1.))
        key_w, _ = jr.split(key)
        self.weight = jr.truncated_normal(
            key_w, shape=(out_size, in_size), lower=-2., upper=2.
        ) * lim
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: XArray) -> XArray:
        return self.weight @ x + self.bias


class ResidualNetwork(eqx.Module):
    _in: Linear
    layers: tuple[Linear]
    dropouts: tuple[eqx.nn.Dropout]
    _out: Linear
    activation: Callable
    q_dim: Optional[int] = None
    a_dim: Optional[int] = None
    t1: float

    @typecheck
    def __init__(
        self, 
        in_size: int, 
        width_size: int, 
        depth: Optional[int], 
        q_dim: Optional[int] = None,
        a_dim: Optional[int] = None, 
        activation: Callable = jax.nn.tanh,
        dropout_p: float = 0.,
        t1: float = 1.,
        t_embedding_dim: int = 1,
        *, 
        key: PRNGKeyArray
    ):
        in_key, *net_keys, out_key = jr.split(key, 2 + depth)

        _in_size = in_size + t_embedding_dim # TODO: time embedding
        if q_dim is not None:
            _in_size += q_dim
        if a_dim is not None:
            _in_size += a_dim

        _width_size = width_size + t_embedding_dim # TODO: time embedding
        if q_dim is not None:
            _width_size += q_dim
        if a_dim is not None:
            _width_size += a_dim

        self._in = Linear(_in_size,width_size, key=in_key)
        layers = [
            Linear(_width_size, width_size, key=_key)
            for _key in net_keys 
        ]
        self._out = Linear(width_size, in_size, key=out_key)
        self.layers = tuple(layers)

        dropouts = [
            eqx.nn.Dropout(p=dropout_p) for _ in layers
        ]
        self.dropouts = tuple(dropouts)

        self.activation = activation
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.t1 = t1
    
    def __call__(
        self, 
        t: Scalar, 
        x: XArray, 
        q: Optional[Array] = None, 
        a: Optional[Array] = None, 
        *, 
        key: Optional[PRNGKeyArray] = None
    ) -> XArray:
        t = jnp.atleast_1d(t / self.t1)

        _qa = []
        if q is not None and self.q_dim is not None:
            _qa.append(q)
        if a is not None and self.a_dim is not None:
            _qa.append(a)

        xqat = jnp.concatenate([x, t] + _qa)
        
        h0 = self._in(xqat)
        h = h0
        for l, d in zip(self.layers, self.dropouts):
            # Condition on time at each layer
            hqat = jnp.concatenate([h, t] + _qa)
            h = l(hqat)
            h = d(h, key=key)
            h = self.activation(h)
        o = self._out(h0 + h)
        return o