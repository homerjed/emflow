
from typing import Callable, Optional, Union
from functools import partial
import jax
import jax.numpy as jnp

from custom_types import (
    XArray, XCovariance, YArray, YCovariance, 
    TCovariance, Scalar, typecheck
)
from rf import RectifiedFlow, velocity_to_score
from utils import maybe_clip


"""
    Posterior sampling 
"""


def value_and_jacfwd(f: Callable, x: XArray, **kwargs) -> tuple[XArray, XCovariance]:
    _fn = lambda x: f(x, **kwargs)
    J_fn = partial(jax.jvp, _fn, (x,)) # NOTE: J[E[x|x_t]] w.r.t. x_t 
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, J = jax.vmap(J_fn, out_axes=(None, 1))((basis,))
    return y, J


@typecheck
def get_E_x_x_t_gaussian(
    x_t: XArray, 
    alpha_t: Scalar,
    cov_t: TCovariance, 
    mu_x: XArray,
    inv_cov_x: XCovariance
) -> XArray: 
    # Convert score to expectation via Tweedie; x_t + cov_t * score[p(x_t)]
    # xt - cov_t * (self.cov_x + cov_t).solve(xt - self.mu_x)
    return (x_t + cov_t @ inv_cov_x @ (x_t - mu_x)) / maybe_clip(alpha_t)  # Tweedie with score of analytic G[x|mu_x, cov_x]


@typecheck
def get_E_x_x_t(
    x_t: XArray,
    flow: RectifiedFlow,
    t: Scalar
) -> XArray: 
    # Get E[x|x_t] from RF E[x_1 - x_0|x_t]; Eq 55, 
    # Appendix B.1 (E[x|x_t]=x_t-t*E[x_1-x|x_t])
    return x_t - t * flow.v(t, x_t)


@typecheck
def get_cov_t(flow: RectifiedFlow, t: Scalar) -> TCovariance:
    # Calculate the covariance of p(x_t|x) = G[x_t|alpha_t * x, Sigma_t] 
    (dim,) = flow.x_shape 
    var_t = maybe_clip(jnp.square(flow.sigma(t))) 
    cov_t = jnp.identity(dim) * var_t
    return cov_t


@typecheck
def get_score_y_x(
    y_: YArray, # Data
    x: XArray, # x_t
    t: Scalar, 
    flow: RectifiedFlow,
    cov_y: YCovariance,
    return_score_x: bool = False
) -> Union[XArray, tuple[XArray, XArray]]:
    # Calculate score[p(y|x_t)] with flow model parameterising q(x|x_t) in 
    # q(y|x_t) = int dx p(y|x) q(x|x_t).
    # > Score of Gaussian linear data-likelihood G[y|x, cov_y] 
    #   when using flow matching model for E[x|x_t] = x_t - t * E[x_1 - x|x_t]
    # > Using V[x|x_t] = cov_t @ Jacobian[E[x|x_t]]

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    score_x = velocity_to_score(flow, t, x)

    E_x_x_t, J_E_x_x_t = value_and_jacfwd(get_E_x_x_t, x, flow=flow, t=t)

    V_x_x_t = jnp.matmul(cov_t, J_E_x_x_t) / maybe_clip(alpha_t) # Eq 5 https://arxiv.org/pdf/2310.06721v3
    V_y_x_t = cov_y + V_x_x_t 

    # NOTE: is there supposed to be a V_y_x_t in front of this? Eq 11 of https://arxiv.org/pdf/2310.06721v3
    score_y_x = jnp.linalg.multi_dot(
        [J_E_x_x_t, jnp.linalg.inv(V_y_x_t), y_ - E_x_x_t] # Eq 20, EMdiff
    )

    if return_score_x:
        return score_y_x, score_x
    else:
        return score_y_x


@typecheck
def get_score_gaussian_y_x(
    y_: YArray, 
    x: XArray, # x_t
    t: Scalar,
    flow: RectifiedFlow,
    cov_y: YCovariance,
    mu_x: XArray,
    inv_cov_x: XCovariance 
) -> XArray:

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    E_x_x_t, J_E_x_x_t = value_and_jacfwd(
        get_E_x_x_t_gaussian, 
        x, 
        alpha_t=alpha_t, 
        cov_t=cov_t, 
        mu_x=mu_x, 
        inv_cov_x=inv_cov_x
    ) 

    V_x_x_t = jnp.matmul(cov_t, J_E_x_x_t) / maybe_clip(alpha_t) # Approximation to Eq 21, see Eq 22. (or heuristics; cov_t, inv(cov_t)...)
    V_y_x_t = cov_y + V_x_x_t
    
    score_y_x = jnp.linalg.multi_dot(
        [J_E_x_x_t, jnp.linalg.inv(V_y_x_t), y_ - E_x_x_t] # Eq 20
    )

    return score_y_x


@typecheck
def get_score_gaussian_x_y(
    y_: YArray, 
    x: XArray, # x_t
    t: Scalar,
    flow: RectifiedFlow,
    cov_y: YCovariance,
    mu_x: XArray,
    inv_cov_x: XCovariance
) -> XArray:

    alpha_t = flow.alpha(t)
    cov_t = get_cov_t(flow, t)

    # Tweedie with score of analytic G[x|mu_x, cov_x]
    # score_x = (x + cov_t @ inv_cov_x @ (x - mu_x)) / maybe_clip(alpha_t) 
    score_x = (x + jnp.dot(cov_t @ inv_cov_x, x - mu_x)) / maybe_clip(alpha_t) 

    score_y_x = get_score_gaussian_y_x(
        y_, x, t, flow, cov_y, mu_x, inv_cov_x
    )

    score_x_y = score_y_x + score_x

    return score_x_y


@typecheck
def get_score_y_x_cg(
    y_: YArray, 
    x: XArray, 
    t: Scalar, 
    flow: RectifiedFlow,
    cov_x: XCovariance,
    cov_y: YCovariance,
    *,
    max_iter: int = 5,
    tol: float = 1e-5, 
    return_score_x: bool = False
) -> XArray:

    cov_t = get_cov_t(flow, t)

    x, vjp = jax.vjp(lambda x_t: velocity_to_score(flow, t, x_t), x) # This shouldn't be score?

    y = x # If no A, x is E[x|x_t]?

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
        return x + cov_t @ score, score # divide by alpha_t
    else:
        return x + cov_t @ score