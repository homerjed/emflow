from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from utils import exists, maybe_clip

typecheck = jaxtyped(typechecker=typechecker)

XArray = Float[Array, "2"]

SDEType = Literal["non-singular", "zero-ends", "singular", "gamma"]


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


def get_timestep_embedding(embedding_dim: int) -> Callable[[Scalar], Float[Array, "e"]]:
    def embedding(t: Scalar) -> Float[Array, "e"]:
        # Convert scalar timesteps to an array NOTE: do these arrays get optimised?
        assert embedding_dim % 2 == 0
        if jnp.isscalar(t):
            t = jnp.asarray(t)
        # t *= 1000.
        # half_dim = int(embedding_dim / 2)
        # emb = jnp.log(10_000.) / (half_dim - 1.)
        # emb = jnp.exp(jnp.arange(half_dim) * -emb)
        # emb = t * emb
        # emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])

        freqs = jnp.linspace(0., 1., int(embedding_dim / 2))
        freqs = (1. / 1.e4) ** freqs
        emb = jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)])
        return emb
    return embedding


"""
    Rectified Flow
"""


class RectifiedFlow(eqx.Module):
    net: eqx.Module
    time_embedder: Callable
    x_shape: tuple[int]
    t0: float
    dt0: float
    t1: float
    solver: dfx.AbstractSolver

    @typecheck
    def __init__(
        self, 
        net: eqx.Module, 
        time_embedder: Callable,
        *,
        x_shape: tuple[int],
        t0: float,
        dt0: float,
        t1: float,
        solver: dfx.AbstractSolver = dfx.Euler()
    ):
        self.net = net
        self.time_embedder = time_embedder
        self.x_shape = x_shape
        self.t0 = t0
        self.dt0 = dt0
        self.t1 = t1
        self.solver = solver

    @property
    def soln_kwargs(self) -> dict:
        return dict(
            t0=self.t0, t1=self.t1, dt0=self.dt0, solver=self.solver
        )

    @typecheck
    def alpha(self, t: Scalar) -> Scalar:
        return jnp.maximum(1. - t, 1e-5) # 1. - t #self.t1 - t 

    @typecheck
    def sigma(self, t: Scalar) -> Scalar:
        return jnp.maximum(t, 1e-5) # t * (EPS - 1) - EPS

    @typecheck
    def p_t(self, x_0: XArray, t: Scalar, eps: XArray) -> XArray:
        return self.alpha(t) * x_0 + self.sigma(t) * eps # NOTE: add eps to sigma(t) here so x_t=0 = x_0 + deps

    # @typecheck
    def sde(
        self, 
        x: XArray, 
        t: Scalar, 
        *,
        alpha: float = 1., # NOTE: not alpha_t
        return_score: bool = False,
        sde_type: SDEType = "zero-ends"
    ) -> Union[
        tuple[XArray, Scalar], tuple[XArray, Scalar, XArray]
    ]:
        assert sde_type in ["non-singular", "zero-ends", "singular", "gamma"], "sde_type={}".format(sde_type)

        v = self.v(t, x)
        score = velocity_to_score(flow=None, t=t, x=x, velocity=v)

        # Non-singular SDE[v(x, t)]
        if sde_type == "non-singular":
            f_, g_ = t, t
        # Zero-ends SDE[v(x, t)]
        if sde_type == "zero-ends":
            f_, g_ = t * (1. - t), t * (1. - t)
        # Singular SDE[v(x, t)] 
        if sde_type == "singular":
            f_, g_ = t / (1. - t), t / (1. - t) 
        # if sde_type == "gamma":
        #     # Choosing g~(t) in Eq 12
        #     f_, g_ = 
        
        drift = v + 0.5 * f_ * score * jnp.square(alpha) # NOTE: try working with only velocities instead of scores?
        diffusion = alpha * jnp.sqrt(g_)

        if return_score:
            return drift, diffusion, score
        else:
            return drift, diffusion

    # @typecheck
    def reverse_ode(
        self, 
        x: XArray, 
        t: Scalar, 
        score: Optional[XArray], 
        *,
        alpha: float = 1., # NOTE: not alpha_t
        sde_type: SDEType = "zero-ends"
    ) -> XArray:
        drift, diffusion = self.sde(
            x, t, alpha=alpha, sde_type=sde_type
        ) 
        return drift - 0.5 * jnp.square(diffusion) * score # Posterior score

    @typecheck
    def v(self, t: Scalar, x: XArray) -> XArray:
        t = self.time_embedder(t)
        return self.net(t, x)

    @typecheck
    def __call__(self, t: Scalar, x: XArray) -> XArray:
        return self.v(t, x)


"""
    Sampling utils 
"""


@typecheck
def velocity_to_score(
    flow: Optional[RectifiedFlow], 
    t: Scalar, 
    x: XArray, 
    velocity: Optional[XArray] = None
) -> XArray:
    # Convert velocity predicted by flow to score of its SDE 
    assert not ((velocity is not None) and (flow is not None))
    v = flow.v(t, x) if exists(flow) else velocity
    return (-(1. - t) * v - x) / maybe_clip(t) #maybe_clip(t, EPS) # https://arxiv.org/pdf/2410.02217v1, Eq. 55


@typecheck
def score_to_velocity(
    score: Optional[XArray],
    t: Scalar, 
    x: XArray 
) -> XArray:
    # Convert score associated with velocity of flow
    return -(score * t + x) / maybe_clip(1. - t) 


def get_flow_soln_kwargs(flow: RectifiedFlow, reverse: bool = False) -> dict:
    soln_kwargs = flow.soln_kwargs.copy()
    if reverse:
        soln_kwargs["t1"], soln_kwargs["t0"] = soln_kwargs["t0"], soln_kwargs["t1"]
        soln_kwargs["dt0"] = -soln_kwargs["dt0"]
    return soln_kwargs