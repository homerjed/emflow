from typing import Callable, Literal
from jaxtyping import PRNGKeyArray, Float, Array, Scalar, jaxtyped
from beartype import beartype as typechecker


typecheck = jaxtyped(typechecker=typechecker)

YArray = Float[Array, "d"] | Float[Array, "_ _ _"]

YCovariance = Float[Array, "d d"] 

XArray = Float[Array, "d"] | Float[Array, "_ _ _"]

XCovariance = Float[Array, "d d"]

TCovariance = Float[Array, "d d"] 

XSampleFn = Callable[[PRNGKeyArray], XArray]

XYSampleFn = Callable[[PRNGKeyArray, XArray], XArray]
 
SDEType = Literal["non-singular", "zero-ends", "singular", "gamma"]

SampleType = Literal["ddim", "sde", "ode"]

Datasets = Literal["gmm", "moons", "blob", "double-blob", "mnist"]