import math
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar

from custom_types import XArray, SDEType, typecheck
from utils import exists, maybe_clip
from resnet import ResidualNetwork
from dit import DiT


def identity(t: Scalar) -> Scalar:
    return t


def cosine_time(t: Scalar) -> Scalar:
    return 1. - (1. / (jnp.tan(0.5 * jnp.pi * t) + 1.)) # t1?


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
    time_embedder: Optional[Callable]
    x_shape: tuple[int]
    x_dim: int
    t0: float
    dt0: float
    t1: float
    solver: dfx.AbstractSolver

    @typecheck
    def __init__(
        self, 
        net: ResidualNetwork | DiT | eqx.Module, 
        time_embedder: Optional[Callable],
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
        self.x_dim = math.prod(x_shape)
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
        return jnp.maximum(1. - t, 1e-5) 

    @typecheck
    def sigma(self, t: Scalar) -> Scalar:
        return jnp.maximum(t, 1e-5) 

    @typecheck
    def cov(self, t: Scalar) -> Float[Array, "_ _"]:
        return jnp.identity(self.x_dim) * jnp.square(jnp.maximum(t, 1e-5))

    @typecheck
    def p_t(self, x_0: XArray, t: Scalar, eps: XArray) -> XArray:
        return self.alpha(t) * x_0 + self.sigma(t) * eps # NOTE: add eps to sigma(t) here so x_t=0 = x_0 + deps

    @typecheck
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

        # Non-singular SDE[v(x, t)]; NOTE: at ends of time, f and g are zero, so score is multipled by zero?
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

    @typecheck
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
        if exists(self.time_embedder):
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


def get_rectified_flow(model_config: dict, key: PRNGKeyArray) -> RectifiedFlow:

    assert model_config.model_type in ["resnet", "dit"]

    if model_config.model_type == "resnet":
        time_embedder = get_timestep_embedding(
            model_config.time_embedding_dim
        )

        net = ResidualNetwork(
            model_config.data_dim, 
            width_size=model_config.width_size, 
            depth=model_config.depth, 
            t_embedding_dim=model_config.time_embedding_dim, 
            t1=model_config.soln_kwargs["t1"],
            key=key
        )

        x_shape = (model_config.data_dim,)

    if model_config.model_type == "dit":
        time_embedder = None

        net = DiT(
            model_config.img_size,
            patch_size=model_config.patch_size,
            channels=model_config.channels,
            embed_dim=model_config.embed_dim,
            depth=model_config.depth,
            n_heads=model_config.n_heads,
            key=key
        )

        x_shape = (model_config.img_size ** 2,)

    flow = RectifiedFlow(
        net, 
        time_embedder, # Not needed for DiT
        x_shape=x_shape,
        **model_config.soln_kwargs
    )
    return flow