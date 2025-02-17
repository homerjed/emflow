import time
import os
from copy import deepcopy
from functools import partial
from shutil import rmtree
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
import optax
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, PyTree, jaxtyped
from beartype import beartype as typechecker

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange
import tensorflow_probability.substrates.jax.distributions as tfd

jax.config.update("jax_debug_nans", True)
jax.config.update('jax_disable_jit', True)

from utils import ppca
from soap import soap

typecheck = jaxtyped(typechecker=typechecker)

XArray = Float[Array, "2"]

Covariance = Float[Array, "2 2"]

XSampleFn = Callable[[PRNGKeyArray], XArray]

XYSampleFn = Callable[[PRNGKeyArray, XArray], XArray]

EPS = 1e-10 # Stop diving by zero with alpha_t, 1 - t, ...


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


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


def get_timestep_embedding(t: Scalar, embedding_dim: int) -> Float[Array, "e"]:
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
        return 1. - t #self.t1 - t 

    @typecheck
    def sigma(self, t: Scalar) -> Scalar:
        return t

    @typecheck
    def p_t(self, x_0: XArray, t: Scalar, eps: XArray) -> XArray:
        return self.alpha(t) * x_0 + self.sigma(t) * eps

    @typecheck
    def sde(
        self, 
        x: XArray, 
        t: Scalar, 
        alpha: float = 1., # NOTE: not alpha_t
        return_score: bool = False,
        sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends"
    ) -> Union[
        tuple[XArray, Scalar], tuple[XArray, Scalar, XArray]
    ]:
        v = self.v(t, x)
        score = velocity_to_score(flow=None, t=t, x=x, velocity=v)

        # Non-singular SDE[v(x, t)]
        if sde_type == "non-singular":
            # drift = v + 0.5 * t * score * jnp.square(alpha) 
            # diffusion = alpha * jnp.sqrt(t)
            f_, g_ = t, t
        # Zero-ends SDE[v(x, t)]
        if sde_type == "zero-ends":
            # drift = v + 0.5 * t * (1. - t) * score * jnp.square(alpha) 
            # diffusion = alpha * jnp.sqrt(t * (1. - t))
            f_, g_ = t * (1. - t), t * (1. - t)
        # Singular SDE[v(x, t)] 
        if sde_type == "singular":
            # drift = v + 0.5 * t / (1. - t) * score * jnp.square(alpha) 
            # diffusion = alpha * jnp.sqrt(t / (1. - t))
            f_, g_ = t / (1. - t), t / (1. - t) 
        
        drift = v + 0.5 * f_ * score * jnp.square(alpha)
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
        sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends"
    ) -> XArray:
        # NOTE: possible bug with return score
        drift, diffusion = self.sde(
            x, t, alpha, return_score=False, sde_type=sde_type
        ) 
        return drift - 0.5 * jnp.square(diffusion) * score # Posterior score

    @typecheck
    def mu_sigma_t(self, t: Scalar) -> tuple[Scalar, Scalar]:
        return self.alpha(t), self.sigma(t) 

    @typecheck
    def v(self, t: Scalar, x: XArray) -> XArray:
        t = self.time_embedder(t)
        return self.net(t, x)

    @typecheck
    def __call__(self, t: Scalar, x: XArray) -> XArray:
        return self.v(t, x)


"""
    ODE sampling (with score[p(y|x)])
"""


@typecheck
def velocity_to_score(
    flow: Optional[RectifiedFlow], 
    t: Scalar, 
    x: XArray, 
    velocity: Optional[XArray] = None
) -> XArray:
    # Convert velocity predicted by flow to score of its SDE 
    assert not ((velocity is not None) and flow is not None)
    v = flow.v(t, x) if exists(flow) else velocity
    return (-(1. - t) * v - x) / maybe_min(t, EPS) # https://arxiv.org/pdf/2410.02217v1, Eq. 55


@typecheck
def score_to_velocity(
    score: Optional[XArray],
    t: Scalar, 
    x: XArray, 
) -> XArray:
    # Convert score associated with velocity of flow
    return -(score * t + x) / maybe_min(1. - t, EPS) 


def get_flow_soln_kwargs(flow: RectifiedFlow, reverse: bool = False):
    soln_kwargs = flow.soln_kwargs.copy()
    if reverse:
        soln_kwargs["t1"], soln_kwargs["t0"] = soln_kwargs["t0"], soln_kwargs["t1"]
        soln_kwargs["dt0"] = -soln_kwargs["dt0"]
    return soln_kwargs


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
    loss_type: Literal["mse", "huber"] = "mse",
    time_schedule: Optional[Callable[[Scalar], Scalar]] = cosine_time
) -> tuple[Scalar, Scalar]:
    """
        Computes MSE between the conditional vector field (x1 - x0)
        and a vector field given by the neural network.
        - NOTE: use EMA?
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


"""
    Data
"""


def get_data(key: PRNGKeyArray, n: int) -> Float[Array, "n 2"]:
    seed = int(jnp.sum(jr.key_data(key)))
    # X, _ = make_moons(n, noise=0.04, random_state=seed)

    N = 8 # Mixture components
    alphas = jnp.linspace(0, 2. * jnp.pi * (1. - 1. / N), N)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    means = jnp.stack([xs, ys], axis=1)

    gaussian_mixture = tfd.Mixture(
        cat=tfd.Categorical(
            probs=jnp.ones((means.shape[0])) / means.shape[0]
        ),
        components=[
            tfd.MultivariateNormalDiag(
                loc=mu, scale_diag=jnp.ones_like(mu) * 0.1
            )
            for mu in means
        ]
    )
    X = gaussian_mixture.sample((n,), seed=key)

    # s = StandardScaler() # Need to keep doing this for each EM's x|y?
    # X = s.fit_transform(X)
    return jnp.asarray(X)


def measurement(key: PRNGKeyArray, x: XArray, cov_y: Covariance) -> XArray: 
    # Sample from G[y|x, cov_y]
    return jr.multivariate_normal(key, x, cov_y) 


"""
    Posterior sampling 
"""


def maybe_min(v, eps):
    return jnp.minimum(v, eps) if exists(eps) else v


@typecheck
def get_score_y_x_cg(
    y_: XArray, 
    x: XArray, 
    t: Scalar, 
    flow: RectifiedFlow,
    cov_x: Covariance,
    cov_y: Covariance,
    *,
    max_iter: int = 5,
    tol: float = 1e-5, 
    return_score_x: bool = False
) -> XArray:

    cov_t = jnp.identity(x.size) * jnp.square(flow.sigma(t)) 

    x, vjp = jax.vjp(lambda x_t: velocity_to_score(flow, t, x_t), x) # This shouldn't be score?
    y = x # If no A, x is E[x|x_t]

    # Get linear operator Mv = b to solve for v given b, choosing heuristic for V[x|x_t]
    if cov_x is None:
        cov_y_xt = lambda v: cov_y @ v + cov_t * vjp(v) # Is this right?
    else:
        cov_x_xt = cov_t + (-(cov_t ** 2.)) * jnp.linalg.inv(cov_x + cov_t)
        cov_y_xt = lambda v: cov_y @ v + cov_x_xt @ v

    b = y_ - y # y_ is data

    # This is a linear operator in lineax?
    v, _ = jax.scipy.sparse.linalg.cg(
        A=cov_y_xt, b=b, tol=tol, maxiter=max_iter
    )

    (score,) = vjp(v) 

    if return_score_x:
        return x + cov_t @ score, score
    else:
        return x + cov_t @ score


@typecheck
def value_and_jacfwd(
    f: Callable[[XArray, Scalar, Covariance, XArray], XArray], 
    x: XArray, 
    alpha: Scalar, 
    cov: Covariance, 
    score: XArray
) -> tuple[XArray, Covariance]:
    J_fn = partial(jax.jvp, lambda x: f(x, alpha, cov, score), (x,)) # NOTE: J[E[x|x_t]] w.r.t. x_t 
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, J = jax.vmap(J_fn, out_axes=(None, 1))((basis,))
    return y, J


@typecheck
def get_E_x_x_t(
    x_t: XArray, 
    alpha_t: Scalar,
    cov_t: Covariance, 
    score_t: XArray
) -> XArray: 
    # Convert score to expectation via Tweedie; x_t + cov_t * score[p(x_t)]
    # return x_t / jnp.minimum(alpha_t, EPS) + cov_t @ score_t
    return (x_t + cov_t @ score_t) / maybe_min(alpha_t, EPS)


@typecheck
def get_cov_t(flow: RectifiedFlow, t: Scalar) -> Covariance:
    # Calculate the covariance of p(x_t|x) = G[x_t|alpha_t * x, Sigma_t] 
    (dim,) = flow.x_shape # dim = jnp.prod(jnp.asarray(flow.x_shape)) 
    cov_t = jnp.identity(dim) * jnp.square(flow.sigma(t))
    return cov_t


@typecheck
def get_score_y_x(
    y_: XArray, # Data
    x: XArray, # x_t
    t: Scalar, 
    flow: RectifiedFlow,
    cov_y: Covariance,
    return_score_x: bool = False
) -> Union[XArray, tuple[XArray, XArray]]:
    # Score of Gaussian linear data-likelihood G[y|x, cov_y] 

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    score_x = velocity_to_score(flow, t, x)

    E_x_x_t, J_E_x_x_t = value_and_jacfwd(
        get_E_x_x_t, x, alpha=alpha_t, cov=cov_t, score=score_x
    )

    V_x_x_t = cov_t @ J_E_x_x_t / maybe_min(jnp.square(alpha_t), EPS) # Eq 22
    V_y_x_t = cov_y + V_x_x_t 

    score_y_x = J_E_x_x_t.T @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20

    if return_score_x:
        return score_y_x, score_x
    else:
        return score_y_x


@typecheck
def get_score_gaussian_y_x(
    y_: XArray, 
    x: XArray, # x_t
    t: Scalar,
    flow: RectifiedFlow,
    # alpha_t: Scalar,
    # cov_t: Covariance,
    cov_y: Covariance,
    mu_x: XArray,
    inv_cov_x: XArray
) -> XArray:
    # NOTE: alpha_t correct here?

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    # score_x = x / jnp.minimum(alpha_t, EPS) + cov_t @ inv_cov_x @ (x - mu_x) # Tweedie with score of analytic G[x|mu_x, cov_x]
    score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_min(alpha_t, EPS)  # Tweedie with score of analytic G[x|mu_x, cov_x]

    E_x_x_t, J_E_x_x_t = value_and_jacfwd(
        get_E_x_x_t, x, alpha=alpha_t, cov=cov_t, score=score_x
    ) 

    V_x_x_t = cov_t @ J_E_x_x_t / maybe_min(jnp.square(alpha_t), EPS) # Approximation to Eq 21, see Eq 22. (or heuristics; cov_t, inv(cov_t)...)
    V_y_x_t = cov_y + V_x_x_t
    
    score_y_x = J_E_x_x_t.T @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20
    return score_y_x


@typecheck
def get_score_gaussian_x_y(
    y_: XArray, 
    x: XArray, # x_t
    t: Scalar,
    flow: RectifiedFlow,
    # alpha_t: Scalar,
    # cov_t: Covariance,
    cov_y: Covariance,
    mu_x: XArray,
    inv_cov_x: Covariance
) -> XArray:
    # Score of Gaussian kernel centred on data NOTE: mode="Gaussian" for init

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    # # NOTE: alpha_t correct here?
    # # score_x = x / jnp.minimum(alpha_t, EPS) + cov_t @ inv_cov_x @ (x - mu_x) # Tweedie with score of analytic G[x|mu_x, cov_x]
    score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_min(alpha_t, EPS) # Tweedie with score of analytic G[x|mu_x, cov_x]

    # E_x_x_t, J_E_x_x_t = value_and_jacfwd(
    #     get_E_x_x_t, x, alpha=alpha_t, cov=cov_t, score=score_x
    # ) 

    # V_x_x_t = cov_t @ J_E_x_x_t / maybe_min(jnp.square(alpha_t), EPS)  # Approximation to Eq 21, see Eq 22. (or heuristics; cov_t, inv(cov_t)...)
    # V_y_x_t = cov_y + V_x_x_t
    
    # score_y_x = J_E_x_x_t.T @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20

    score_y_x = get_score_gaussian_y_x(y_, x, t, flow, cov_y, mu_x, inv_cov_x)

    score_x_y = score_y_x + score_x

    return score_x_y


@typecheck
@eqx.filter_jit
def single_x_y_ddim_sample_fn(
    flow: RectifiedFlow,
    key: PRNGKeyArray, 
    y_: XArray,
    cov_y: Covariance, 
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[Covariance] = None, 
    *,
    q_0_sampling: bool = False, # Sampling initially or not
    n_steps: int = 500,
    eta: float = 1., # DDIM stochasticity 
    sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends",
    mode: Literal["full", "cg"] = "full"
) -> XArray:
    # DDIM sampler including data-likelihood score

    key_z, key_sample = jr.split(key)

    # Reversed times
    times = jnp.linspace(
        flow.soln_kwargs["t1"], 
        flow.soln_kwargs["t0"], 
        n_steps + 1 
    )

    flow = eqx.nn.inference_mode(flow)

    def sample_step(i: Scalar, x_t_key: XArray) -> tuple[XArray, PRNGKeyArray]:
        x_t, key = x_t_key

        s, t = times[i], times[i + 1]
        dt = t - s

        if q_0_sampling:
            assert (mu_x is not None) and (inv_cov_x is not None)

            # Implement CG method for this
            score_y_x, score_x = get_score_gaussian_x_y(
                y_, x_t, t, flow, cov_y, mu_x, inv_cov_x
            )
        else:
            if mode == "full":
                score_y_x, score_x = get_score_y_x(
                    y_, x_t, t, flow, cov_y, return_score_x=True # x is x_t
                ) 
            if mode == "cg":
                score_y_x, score_x = get_score_y_x_cg(
                    y_, x_t, t, flow, cov_x, cov_y, return_score_x=True
                ) 

        score_x_y = score_y_x + score_x

        drift, diffusion = flow.sde(z, t, sde_type=sde_type) # NOTE: implement SDE here?! DDIM for arbitrary SDE => use zero ends here 

        # Compute deterministic step for DDIM (reverse ODE)
        deterministic_update = drift - 0.5 * jnp.square(diffusion) * score_x_y

        # Stochasticity term (optional, eta=0 for DDIM)
        noise = jr.normal(key, x_t.shape)
        stochastic_update = eta * diffusion * jnp.sqrt(-dt) * noise # NOTE: abs on dt?

        # Update x_t using DDIM 
        x_s = x_t + deterministic_update * dt + stochastic_update

        return x_s, key

    z = jr.normal(key_z, flow.x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, key_sample)
    )

    return x


@typecheck
@eqx.filter_jit
def single_x_y_sample_fn_ode(
    flow: RectifiedFlow,
    key: PRNGKeyArray, 
    y_: XArray,
    cov_y: Covariance, 
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[Covariance] = None, 
    *,
    mode: Literal["full", "cg"] = "full",
    sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends",
    q_0_sampling: bool = False # Sampling initial q_0(x) or model q(x)
) -> XArray:
    # Latent posterior sampling function

    def reverse_ode(t, x, args):
        # Sampling along conditional score p(x|y)
        t = jnp.asarray(t)

        if q_0_sampling:
            # Implement CG method for this
            # alpha_t = flow.alpha(t)
            # cov_t = get_cov_t(flow, t)
            score_y_x, score_x = get_score_gaussian_x_y(
                y_, x, t, flow, mu_x, inv_cov_x, cov_y
            )
        else:
            if mode == "full":
                score_y_x, score_x = get_score_y_x(
                    y_, x, t, flow, cov_y, return_score_x=True # x is x_t
                ) 
            if mode == "cg":
                score_y_x, score_x = get_score_y_x_cg(
                    y_, x, t, flow, cov_x, cov_y, return_score_x=True
                ) 

        score_x_y = score_x + score_y_x

        return flow.reverse_ode(x, t, score=score_x_y, sde_type=sde_type) 

    sol = dfx.diffeqsolve(
        dfx.ODETerm(reverse_ode), 
        **get_flow_soln_kwargs(flow, reverse=True),
        y0=jr.normal(key, flow.x_shape) 
    )
    return sol.ys[0]


"""
    Stochastic sampling
"""


@typecheck
@eqx.filter_jit
def single_non_singular_sample_fn(
    flow: RectifiedFlow, 
    key: PRNGKeyArray, 
    *,
    g_scale: float = 0.1, 
    n_steps: int = 500,
    n: float = 1., 
    m: float = 0.
) -> XArray:
    """
        Possibly add score callable for p(y|x)?
    """

    key_z, key_sample = jr.split(key)

    t = jnp.linspace(
        flow.soln_kwargs["t0"], flow.soln_kwargs["t1"], n_steps + 1
    )

    flow = eqx.nn.inference_mode(flow)

    def sample_step(i, z):
        z, key = z

        key, key_eps = jr.split(key)

        _t = t[i] 
        _dt = t[i + 1] - t[i]

        eps = jr.normal(key_eps, z.shape)

        z_hat = flow.v(1. - _t, z) # Add velocity(score_y_x) here

        _z_hat = -z_hat
        g = g_scale * jnp.power(_t, 0.5 * n) * jnp.power(1. - _t, 0.5 * m)
        s_u = -((1. - _t) * _z_hat + z)
        fr = _z_hat - jnp.square(g_scale) * jnp.power(_t, n - 1.) * jnp.power(1. - _t, m) * 0.5 * s_u

        dbt = jnp.sqrt(jnp.abs(_dt)) * eps
        z = z + fr * _dt + g * dbt

        return z, key 

    z = jr.normal(key_z, flow.x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, key_sample)
    )

    return x


@typecheck
@eqx.filter_jit
def single_non_singular_x_y_sample_fn(
    flow: RectifiedFlow, 
    key: PRNGKeyArray, 
    y_: XArray,
    cov_y: Covariance,
    mu_x: Optional[XArray] = None,
    inv_cov_x: Optional[Covariance] = None,
    *,
    g_scale: float = 0.1, 
    n_steps: int = 1000,
    n: float = 1., 
    m: float = 0.,
    mode: Literal["full", "cg"] = "full",
    q_0_sampling: bool = False
) -> XArray:
    """
        Stochastic sampling of p(x|y)
    """

    key_z, key_sample = jr.split(key)

    # Reversed time
    t = jnp.linspace(
        flow.soln_kwargs["t1"], flow.soln_kwargs["t0"], n_steps + 1
    )

    flow = eqx.nn.inference_mode(flow)

    def sample_step(i, z):
        z, key = z

        key, key_eps = jr.split(key)

        _t = t[i] 
        _dt = t[i + 1] - t[i]

        eps = jr.normal(key_eps, z.shape)

        if q_0_sampling:
            assert inv_cov_x is not None

            score_p_y_x_t = get_score_gaussian_y_x(
                y_, z, 1. - _t, flow, cov_y, mu_x, inv_cov_x
            )
        else:
            # score_p_y_x_t = get_score_y_x(y_, z, _t, flow, cov_y)
            
            if mode == "full":
                score_p_y_x_t = get_score_y_x(
                    y_, z, 1. - _t, flow, cov_y # x is x_t
                ) 
            if mode == "cg":
                score_p_y_x_t = get_score_y_x_cg(
                    y_, z, 1. - _t, flow, cov_x, cov_y
                ) 

        z_hat = flow.v(1. - _t, z) + score_to_velocity(score_p_y_x_t, 1. - _t, z) # Add velocity(score_y_x) here

        _z_hat = -z_hat
        g = g_scale * jnp.power(_t, 0.5 * n) * jnp.power(1. - _t, 0.5 * m)
        s_u = -((1. - _t) * _z_hat + z)
        fr = _z_hat - jnp.square(g_scale) * jnp.power(_t, n - 1.) * jnp.power(1. - _t, m) * 0.5 * s_u

        dbt = jnp.sqrt(jnp.abs(_dt)) * eps
        z = z + fr * _dt + g * dbt

        return z, key 

    z = jr.normal(key_z, flow.x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, key_sample)
    )

    return x


"""
    Sampler utils
"""


@typecheck
def get_x_y_sampler_ode(
    flow: RectifiedFlow, 
    cov_y: Covariance, 
    mu_x: Optional[XArray] = None, 
    cov_x: Optional[Covariance] = None, 
    *,
    mode: Literal["full", "cg"] = "full",
    sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends",
    q_0_sampling: bool = False
) -> XYSampleFn:

    if exists(cov_x): 
        inv_cov_x = jnp.linalg.inv(cov_x)
    else:
        inv_cov_x = None

    fn = lambda key, y_: single_x_y_sample_fn_ode(
        flow, 
        key, 
        y_, 
        cov_y, 
        mu_x,
        inv_cov_x, 
        mode=mode, 
        sde_type=sde_type,
        q_0_sampling=q_0_sampling # NOTE: Why does this need cov_x? Does cov_x need iterating?
    )
    return fn


@typecheck
def get_x_y_sampler_ddim(
    flow: RectifiedFlow, 
    cov_y: Covariance, 
    mu_x: Optional[XArray] = None, 
    cov_x: Optional[Covariance] = None, 
    *,
    n_steps: int = 1000,
    eta: float = 1., # Stochasticity in DDIM
    mode: Literal["full", "cg"] = "full",
    sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends",
    q_0_sampling: bool = False # Sampling initial q_0(x) or model q(x)
) -> XYSampleFn:

    if exists(cov_x): 
        inv_cov_x = jnp.linalg.inv(cov_x)
    else:
        inv_cov_x = None

    fn = lambda key, y_: single_x_y_ddim_sample_fn(
        flow, 
        key, 
        y_, 
        cov_y=cov_y, 
        mu_x=mu_x,
        inv_cov_x=inv_cov_x, 
        q_0_sampling=q_0_sampling, 
        n_steps=n_steps,
        eta=eta,
        sde_type=sde_type,
        mode=mode # NOTE: Why does this need cov_x? Does cov_x need iterating?
    )
    return fn


@typecheck
def get_x_y_sampler_sde(
    flow: RectifiedFlow, 
    cov_y: Covariance, 
    mu_x: Optional[XArray] = None, 
    cov_x: Optional[Covariance] = None, 
    *,
    n_steps: int = 1000,
    g_scale: float = 0.1,
    n: float = 1.,
    m: float = 0.,
    mode: Literal["full", "cg"] = "full",
    q_0_sampling: bool = False # Sampling initial q_0(x) or model q(x)
) -> XYSampleFn:

    if exists(cov_x): 
        inv_cov_x = jnp.linalg.inv(cov_x)
    else:
        inv_cov_x = None

    fn = lambda key, y_: single_non_singular_x_y_sample_fn(
        flow, 
        key, 
        y_, 
        cov_y=cov_y, 
        mu_x=mu_x,
        inv_cov_x=inv_cov_x, 
        n_steps=n_steps,
        g_scale=g_scale,
        n=n,
        m=m,
        mode=mode,
        q_0_sampling=q_0_sampling, 
    )
    return fn


@typecheck
def get_non_singular_sample_fn(
    flow: RectifiedFlow,
    *,
    n_steps: int = 1000,
    g_scale: float = 0.1,
    n: float = 1.,
    m: float = 0.
) -> XSampleFn:
    # Sampler for e.g. sampling latents from model p(x) without y 
    fn = lambda key: single_non_singular_sample_fn(
        flow, key=key, g_scale=g_scale, n_steps=n_steps, n=n, m=m
    )
    return fn


"""
    Plots & utils
"""


def clear_and_get_results_dir(save_dir: str) -> str:
    # Image save directories
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


def plot_samples(X, X_Y, Y, X_, n_plot=8000, iteration=0, save_dir="imgs/"):
    plot_kwargs = dict(s=0.08, marker=".")
    fig, axs = plt.subplots(2, 1, figsize=(4., 9.), dpi=200)
    ax = axs[0]
    ax.set_title("EM: iteration {}".format(iteration))
    ax.scatter(*X[:n_plot].T, color="k", label=r"$x\sim p(x)$", **plot_kwargs)
    ax.scatter(*X_Y[:n_plot].T, color="b", label=r"$x\sim p_{\theta}(x|y)$", **plot_kwargs) 
    ax.scatter(*Y[:n_plot].T, color="r", label=r"$y\sim p(y|x)$", **plot_kwargs) 
    legend = ax.legend(frameon=False, loc="upper right")
    legend.get_texts()[0].set_color("k") 
    legend.get_texts()[1].set_color("b") 
    legend.get_texts()[2].set_color("r") 
    ax = axs[1]
    ax.scatter(*X[:n_plot].T, color="k", label=r"$x\sim p(x)$", **plot_kwargs) 
    ax.scatter(*X_[:n_plot].T, color="b", label=r"$x\sim p_{\theta}(x)$", **plot_kwargs) 
    legend = ax.legend(frameon=False, loc="upper right")
    legend.get_texts()[0].set_color("k") 
    legend.get_texts()[1].set_color("b") 
    for ax in axs:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
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
            if img.endswith(("png")) and "L" not in img
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
    flow: RectifiedFlow, 
    optimiser: Union[
        optax.GradientTransformation, 
        optax.GradientTransformationExtraArgs
    ] = optax.adamw, 
    lr: float = 1e-3, 
    use_lr_schedule: bool = False, 
    initial_lr: Optional[float] = 1e-6, 
    n_data: Optional[int] = None,
    n_epochs_warmup: Optional[int] = None,
    diffusion_iterations: Optional[int] = None
) -> tuple[
    optax.GradientTransformationExtraArgs, optax.OptState
]:
    if use_lr_schedule:
        n_steps_per_epoch = int(n_data / n_batch)

        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=initial_lr, 
            peak_value=lr, 
            warmup_steps=n_epochs_warmup * n_steps_per_epoch,
            decay_steps=diffusion_iterations * n_steps_per_epoch, 
            end_value=lr
        )

    opt = optimiser(scheduler if exists(scheduler) else lr)
    opt = optax.chain(optax.clip_by_global_norm(1.), opt) 

    opt_state = opt.init(eqx.filter(flow, eqx.is_array))

    return opt, opt_state


def clip_latents(X, x_clip_limit):
    return X[jnp.all(jnp.logical_and(-x_clip_limit < X, X < x_clip_limit), axis=-1)] 


if __name__ == "__main__":
    key = jr.key(int(time.time()))

    save_dir             = clear_and_get_results_dir(save_dir="imgs__/")


    # Train
    em_iterations        = 64
    diffusion_iterations = 1_000
    n_batch              = 5_000
    loss_type            = "mse"
    time_schedule        = identity
    lr                   = 1e-3
    optimiser            = soap
    use_lr_schedule      = True
    initial_lr           = 1e-6
    n_epochs_warmup      = 1
    ppca_pretrain        = False
    n_pca_iterations     = 10
    clip_x_y             = False #True # Clip sampled latents
    x_clip_limit         = 4.
    re_init_opt_state    = True
    n_plot               = 10_000
    sampling_mode        = "sde"        # ODE, SDE or DDIM
    sde_type             = "zero-ends"  # SDE of flow ODE
    mode                 = "full"       # CG mode or not
    use_ema              = False
    ema_rate             = 0.9995

    # Model
    width_size           = 256 #128
    depth                = 2 #5
    activation           = jax.nn.gelu # silu 
    soln_kwargs          = dict(t0=0., dt0=0.1, t1=1., solver=dfx.Euler()) # For ODE
    time_embedding_dim   = 32

    # Data
    data_dim             = 2
    n_data               = 100_000
    sigma_y              = 0.2 # Tiny eigenvalues may have been numerically unstable?
    cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)

    assert sampling_mode in ["ddim", "ode", "sde"]
    assert sde_type in ["non-singular", "zero-ends", "singular"]
    assert mode in ["full", "cg"]

    key_net, key_data, key_measurement, key_ppca, key_em = jr.split(key, 5)

    # Rectified flow model
    time_embedder = partial(
        get_timestep_embedding, embedding_dim=time_embedding_dim
    )

    net = ResidualNetwork(
        data_dim, 
        width_size=width_size, 
        depth=depth, 
        t_embedding_dim=time_embedding_dim, 
        t1=soln_kwargs["t1"],
        key=key_net 
    )

    flow = RectifiedFlow(
        net, time_embedder, x_shape=(data_dim,), **soln_kwargs
    )

    if use_ema:
        ema_flow = deepcopy(flow)

    # opt_state = opt.init(eqx.filter(flow, eqx.is_array))
    opt, opt_state = get_opt_and_state(
        flow, 
        optimiser,
        lr=lr, 
        use_lr_schedule=use_lr_schedule, 
        initial_lr=initial_lr,
        n_epochs_warmup=n_epochs_warmup,
        n_data=n_data,
        diffusion_iterations=diffusion_iterations
    )

    # Latents
    X = get_data(key_data, n_data)

    # Generate y ~ G[y|x, cov_y]
    keys = jr.split(key_measurement, n_data)
    Y = jax.vmap(partial(measurement, cov_y=cov_y))(keys, X) 

    # PPCA pre-training for q_0(x|mu_x, cov_x)
    mu_x = jnp.zeros(data_dim)
    cov_x = jnp.identity(data_dim) # Start at cov_y?
    if ppca_pretrain:

        X_ = Y
        for s in trange(
            n_pca_iterations, desc="PPCA Training", colour="blue"
        ):
            key_pca, key_sample = jr.split(jr.fold_in(key_ppca, s))

            mu_x, cov_x = ppca(X_, key_pca, rank=data_dim)

            if sampling_mode == "ddim":
                sampler = get_x_y_sampler_ddim(
                    flow, cov_y, mu_x, cov_x, sde_type=sde_type, q_0_sampling=True
                )
            if sampling_mode == "ode":
                sampler = get_x_y_sampler_ode(
                    flow, cov_y, mu_x, cov_x, sde_type=sde_type, q_0_sampling=True
                )
            if sampling_mode == "sde":
                sampler = get_x_y_sampler_sde(
                    flow, cov_y, mu_x, cov_x, q_0_sampling=True
                )
            keys = jr.split(key_sample, n_data)
            X_ = jax.vmap(sampler)(keys, Y)

        print("mu/cov x:", mu_x, cov_x)
    else:
        X_ = jax.vmap(partial(measurement, cov_y=2. * cov_y))(keys, X) # Testing

    # Sample latents unconditionally
    X_test = jax.vmap(get_non_singular_sample_fn(flow))(keys)

    # Plot initial samples
    plot_samples(X, X_, Y, X_test, save_dir=save_dir)

    # Expectation maximisation
    losses_k = []
    for k in range(em_iterations):
        key_k, key_sample = jr.split(jr.fold_in(key_em, k))

        # Train on sampled latents
        losses_i = []
        with trange(
            diffusion_iterations, desc="Training", colour="green"
        ) as steps:
            for i in steps:
                key_x, key_step = jr.split(jr.fold_in(key_k, i))

                x = jr.choice(key_x, X_, (n_batch,)) # Make sure always choosing x ~ p(x|y)

                L, flow, key, opt_state = make_step(
                    flow, 
                    x, 
                    key_step, 
                    opt_state, 
                    opt.update, 
                    loss_type=loss_type, 
                    time_schedule=time_schedule
                )

                if use_ema:
                    ema_flow = apply_ema(ema_flow, flow, ema_rate)

                losses_i.append(L)
                steps.set_postfix_str(f"\r {k=:04d} {L=:.3E}")

        # Plot losses
        losses_k += losses_i
        plot_losses(losses_k, k, diffusion_iterations, save_dir=save_dir)

        # Restart optimiser on previously trained score network
        if re_init_opt_state:
            opt, opt_state = get_opt_and_state(
                flow, 
                optimiser,
                lr=lr, 
                use_lr_schedule=use_lr_schedule, 
                initial_lr=initial_lr,
                n_epochs_warmup=n_epochs_warmup,
                n_data=n_data,
                diffusion_iterations=diffusion_iterations
            )

        # Generate latents from q(x|y)
        if sampling_mode == "ddim":
            sampler = get_x_y_sampler_ddim(
                ema_flow if use_ema else flow, cov_y=cov_y, mode=mode
            )
        if sampling_mode == "ode":
            sampler = get_x_y_sampler_ode(
                ema_flow if use_ema else flow, cov_y=cov_y, sde_type=sde_type, mode=mode
            )
        if sampling_mode == "sde":
            sampler = get_x_y_sampler_sde(
                ema_flow if use_ema else flow, cov_y=cov_y, mode=mode
            )
        keys = jr.split(key_sample, n_data)
        X_ = jax.vmap(sampler)(keys, Y)

        if clip_x_y:
            X_ = clip_latents(X_, x_clip_limit) 

        # Generate latents from q(x)
        X_test = jax.vmap(get_non_singular_sample_fn(ema_flow if use_ema else flow))(keys)

        # Plot latents
        plot_samples(X, X_, Y, X_test, iteration=k + 1, save_dir=save_dir)

        if k > 1:
            create_gif(save_dir)

        # ...automatically begins next k-iteration with parameters from SGM this iteration

# score_y_x = J_E_x_x_t.T @ jnp.linalg.inv(cov_y + cov_t @ J_E_x_x_t) @ (y_ - E_x_x_t) # Eq 20, 22
# return J_E_x_x_t.T @ jnp.linalg.inv(cov_y + V_x_x_t) @ (y_ - E_x_x_t) # Eq 20

    # opt                  = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-3))
    # opt                  = optax.chain(optax.clip_by_global_norm(1.), soap(1e-3)) # NOTE: schedule?


    # @typecheck
    # def sample(
    #     self, 
    #     key: PRNGKeyArray, 
    #     soln_kwargs: Optional[dict] = None
    # ) -> Array:
    #     return single_sample_fn(
    #         self, key, self.x_shape, **default(soln_kwargs, self.soln_kwargs)
    #     )

    # @typecheck
    # def sample_with_score(
    #     self, 
    #     key: PRNGKeyArray, 
    #     y_x_score: Array, 
    #     soln_kwargs: Optional[dict] = None
    # ) -> Array:
    #     # xy_score = score[p(y|x)]
    #     return single_score_sample_fn(
    #         self, 
    #         key, 
    #         self.x_shape, 
    #         y_x_score=y_x_score, 
    #         **default(soln_kwargs, self.soln_kwargs)
    #     )

    # @typecheck
    # def sample_sde(
    #     self,
    #     key: PRNGKeyArray, 
    #     *,
    #     t0: float, 
    #     t1: float, 
    #     g_scale: float = 0.1, 
    #     n_steps: int = 1000,
    #     n: int = 1, 
    #     m: int = 0
    # ) -> Array:
    #     return single_non_singular_sample_fn(
    #         self.v, 
    #         q=None, 
    #         a=None, 
    #         key=key, 
    #         x_shape=self.x_shape, 
    #         t0=default(t0, self.t0), 
    #         t1=default(t1, self.t1), 
    #         n_steps=n_steps,
    #         g_scale=g_scale,
    #         m=m,
    #         n=n
    #     )



# OLD INITIAL SMAPLERS

# @typecheck
# def get_initial_x_y_sampler_ode(
#     flow: RectifiedFlow, 
#     mu_x: XArray, 
#     cov_x: Covariance,
#     cov_y: Covariance
# ) -> Callable[[PRNGKeyArray, XArray], XArray]:
#     fn = lambda key, y_: sample_initial_score_ode(
#         mu_x, cov_x, cov_y, flow, key, y_
#     )
#     return fn


# @typecheck
# def get_initial_x_y_sampler_ode(
#     flow: RectifiedFlow, 
#     mu_x: XArray, 
#     cov_x: Covariance,
#     cov_y: Covariance,
#     *,
#     sde_type: Literal["non-singular", "zero-ends", "singular"] = "non-singular"
# ) -> Callable[[PRNGKeyArray, XArray], XArray]:
#     fn = lambda key, y_: sample_initial_score_ode(
#         mu_x, cov_x, cov_y, flow, key, y_, sde_type=sde_type
#     )
#     return fn


# @typecheck
# def get_initial_x_y_sampler_ddim(
#     flow: RectifiedFlow, 
#     mu_x: XArray, 
#     cov_x: Covariance,
#     cov_y: Covariance,
#     mode: Literal["full", "cg"] = "full"
# ) -> Callable[[PRNGKeyArray, XArray], XArray]:
#     fn = lambda key, y_: single_x_y_ddim_sample_fn(
#         mu_x, cov_x, cov_y, flow, key, y_, q_0_sampling=True, mode=mode
#     )
#     return fn


# @typecheck
# @eqx.filter_jit
# def sample_initial_score_ode(
#     mu_x: XArray, 
#     cov_x: Covariance, 
#     cov_y: Covariance,
#     flow: RectifiedFlow, 
#     key: PRNGKeyArray, 
#     y_: XArray,
#     *,
#     sde_type: Literal["non-singular", "zero-ends", "singular"] = "zero-ends",
# ) -> XArray:
#     # Sample from initial q_0(x|y) with PPCA prior q_0(x)

#     inv_cov_x = jnp.linalg.inv(cov_x)

#     @typecheck
#     def get_score_gaussian_y_x(
#         y_: XArray, 
#         x: XArray, # x_t
#         alpha_t: Scalar,
#         cov_t: Covariance,
#         score: XArray, 
#         cov_y: Covariance
#     ) -> XArray:
#         # Score of Gaussian kernel centred on data NOTE: mode="Gaussian" for init
#         E_x_x_t, J_E_x_x_t = value_and_jacfwd(get_E_x_x_t, x, alpha_t, cov_t, score) # Tweedie; mean and jacobian
#         V_x_x_t = cov_t @ J_E_x_x_t # Or heuristics; cov_t, inv(cov_t)...
#         V_y_x_t = cov_y + V_x_x_t
#         return J_E_x_x_t.T @ jnp.linalg.inv(V_y_x_t) @ (y_ - E_x_x_t) # Eq 20

#     def reverse_ode(t, x, args):
#         # Sampling along conditional score p(x|y)
#         t = jnp.asarray(t)

#         alpha_t = flow.alpha(t)
#         cov_t = get_cov_t(flow, t)

#         # score_x = x / jnp.minimum(alpha_t, EPS) + cov_t @ inv_cov_x @ (x - mu_x) # Tweedie with score of analytic G[x|mu_x, cov_x]
#         score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_min(alpha_t, EPS) # Tweedie with score of analytic G[x|mu_x, cov_x]

#         score_y_x = get_score_gaussian_y_x(y_, x, alpha_t, cov_t, score_x, cov_y) # This y is x_t?

#         score_x_y = score_x + score_y_x

#         return flow.reverse_ode(x, t, score=score_x_y, sde_type=sde_type) 

#     sol = dfx.diffeqsolve(
#         dfx.ODETerm(reverse_ode), 
#         **get_flow_soln_kwargs(flow, reverse=True),
#         y0=jr.normal(key, flow.x_shape) # y1
#     )
#     return sol.ys[0]




# @eqx.filter_jit
# def single_score_sample_fn(
#     flow: RectifiedFlow, 
#     key: PRNGKeyArray, 
#     y_x_score: Optional[XArray] = None,
# ) -> XArray:
#     flow = eqx.nn.inference_mode(flow, True)

#     def _flow(t, x, args):
#         score_x = velocity_to_score(flow, t, x) #(-(1. - t) * flow.v(t, x) - x) / t
#         if y_x_score is not None:
#             score_x_y = score_x + y_x_score
#         return score_x_y
    
#     y1 = jr.normal(key, flow.x_shape)

#     term = dfx.ODETerm(_flow) 
#     sol = dfx.diffeqsolve(
#         term, 
#         **get_flow_soln_kwargs(flow, reverse=True),
#         y0=y1
#     )
#     (y0,) = sol.ys

#     return y0

