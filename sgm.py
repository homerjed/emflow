from typing import Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

typecheck = jaxtyped(typechecker=typechecker)

Covariance = Float[Array, "2 2"]

XArray = Float[Array, "2"]

DEFAULT_SOLN_KWARGS = dict(t0=0., dt=0.01, t1=1., solver=dfx.Heun())

 
def sigma_fn(t):
    return t ** 2. 


def beta_integral(t):
    return t


class VE(eqx.Module):
    sigma_fn: Callable

    def __init__(self, sigma_fn):
        self.sigma_fn = sigma_fn # VE SDE: dx = sqrt(d[sigma^2(t)]/dt)

    def p_t(self, x, t):
        std = jnp.sqrt(jnp.square(self.sigma_fn(t)) - jnp.square(self.sigma_fn(0.))) 
        return x, std

    def sde(self, x, t):
        drift = jnp.zeros_like(x)
        _, dsigma2dt = jax.jvp(
            lambda t: jnp.square(self.sigma_fn(t)), 
            (t,), 
            (jnp.ones_like(t),),
        )
        diffusion = jnp.sqrt(dsigma2dt)
        return drift, diffusion

    def weight(self, t):
        drift, diffusion = self.sde(jnp.zeros((1,)), t)
        return jnp.square(diffusion)

    def reverse_ode(self, score, x, t):
        drift, diffusion = self.sde(x, t)
        return drift - 0.5 * jnp.square(diffusion) * score


class VP(eqx.Module):
    beta_integral_fn: Callable
 
    def __init__(self, beta_integral_fn: Callable):
        self.beta_integral_fn = beta_integral_fn 

    @typecheck
    def beta_fn(self, t: Scalar) -> Scalar:
        _, beta = jax.jvp(
            self.beta_integral_fn, (t,), (jnp.ones_like(t),)
        )
        return beta

    @typecheck
    def p_t(self, x: XArray, t: Scalar) -> tuple[XArray, Scalar]:
        beta_integral = self.beta_integral_fn(t)
        mean = jnp.exp(-0.5 * beta_integral) * x 
        std = jnp.sqrt(-jnp.expm1(-beta_integral)) 
        return mean, std

    @typecheck
    def p_t_sigma_t(self, t: Scalar) -> Scalar:
        beta_integral = self.beta_integral_fn(t)
        std = jnp.sqrt(-jnp.expm1(-beta_integral)) 
        return std

    @typecheck
    def sde(self, x: XArray, t: Scalar) -> tuple[XArray, Scalar]:
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x 
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion
        
    @typecheck
    def weight(self, t: Scalar) -> Scalar:
        return 1. - jnp.exp(-t)

    @typecheck
    def reverse_ode(self, score: XArray, x: XArray, t: Scalar) -> XArray:
        drift, diffusion = self.sde(x, t)
        return drift - 0.5 * jnp.square(diffusion) * score


SDE = VP | VE


class Net(eqx.Module):
    in_size: int
    time_embedding_dim: int = 1
    out_size: int
    layers: list[eqx.Module]

    @typecheck
    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        width_size: int, 
        depth: int, 
        activation: Callable, 
        time_embedding_dim: Optional[int] = None,
        *, 
        key: PRNGKeyArray
    ):
        self.in_size = in_size + time_embedding_dim 
        self.time_embedding_dim = default(time_embedding_dim, 1)
        self.out_size = out_size

        _dims = [
            (_in, _out) 
            for _in, _out in zip(
                [self.in_size] + [width_size] * depth, 
                [width_size] * depth + [out_size]
            )
        ]

        keys = jr.split(key, depth + 1)
        layers = [
           (
               eqx.nn.Linear(_in, _out, key=key),
               activation, 
               eqx.nn.LayerNorm((_out,))

           )
           for (_in, _out), key in zip(_dims, keys)
        ]
        self.layers = layers

    @typecheck
    def __call__(self, x: Float[Array, "{self.in_size}"]) -> Float[Array, "{self.out_size}"]:
        for l, a, n in self.layers[:-1]:
            x = n(a(l(x)))
        l, _, n = self.layers[-1]
        return n(l(x))


def get_timestep_embedding(timesteps: Array, embedding_dim: int):
    # Convert scalar timesteps to an array; this way parameters here are 'frozen'
    assert embedding_dim % 2 == 0
    if jnp.isscalar(timesteps):
        timesteps = jnp.asarray(timesteps) # array
    timesteps *= 1000. # Convert [0, 1] to 'integer' timestep in DDPM
    half_dim = int(embedding_dim / 2)
    emb = jnp.log(10_000.) / (half_dim - 1.)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
    return emb


def exists(v):
    return v is not None
    

class ScoreNet(Net):
    t1: float
    time_embedding_dim: int
    
    def __init__(
        self, 
        t1: float, 
        *args, 
        time_embedding_dim: Optional[int] = None, 
        **kwargs
    ):
        self.t1 = t1
        self.time_embedding_dim = time_embedding_dim
        super().__init__(*args, time_embedding_dim=time_embedding_dim, **kwargs)

    @typecheck
    def __call__(self, t: Scalar, x: XArray) -> XArray:
        t = jnp.atleast_1d(t / self.t1)
        if exists(get_timestep_embedding) and self.time_embedding_dim > 1:
            t = get_timestep_embedding(t, self.time_embedding_dim)
        xt = jnp.concatenate([x, t])
        return super().__call__(xt)


def default(v, d):
    return v if v is not None else d


class SGM(eqx.Module):
    net: ScoreNet 
    sde: SDE
    x_shape: tuple[int]
    soln_kwargs: dict

    @typecheck
    def __init__(
        self, 
        net: ScoreNet, 
        sde: SDE, 
        x_shape: tuple[int], 
        soln_kwargs: Optional[dict] = None
    ):
        self.net = net
        self.sde = sde
        self.x_shape = x_shape
        self.soln_kwargs = default(soln_kwargs, DEFAULT_SOLN_KWARGS)