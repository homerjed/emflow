from typing import Callable, Literal
from jaxtyping import PRNGKeyArray, Float, Array, Scalar, jaxtyped
from beartype import beartype as typechecker


def get_typechecker():
    return jaxtyped(typechecker=typechecker)


YArray = Float[Array, "2"]

YCovariance = Float[Array, "2 2"]

XArray = Float[Array, "2"]

XCovariance = Float[Array, "2 2"]

TCovariance = Float[Array, "2 2"]

XSampleFn = Callable[[PRNGKeyArray], XArray]

XYSampleFn = Callable[[PRNGKeyArray, XArray], XArray]
 
SDEType = Literal["non-singular", "zero-ends", "singular", "gamma"]

SampleType = Literal["ddim", "sde", "ode"]