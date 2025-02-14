from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from sgm import SGM, XArray


@eqx.filter_jit
def single_sample_fn(sgm: SGM, key: PRNGKeyArray) -> XArray:

    def reverse_ode(t, y, args):
        return sgm.sde.reverse_ode(sgm.net(t, y), y, t)

    term = dfx.ODETerm(reverse_ode)
    y1 = jr.normal(key, sgm.x_shape) 
    sol = dfx.diffeqsolve(
        term, 
        sgm.solver_kwargs["solver"],
        sgm.solver_kwargs["t1"],
        sgm.solver_kwargs["t0"],
        -sgm.solver_kwargs["dt"],
        y1
    ) 
    return sol.ys[0]