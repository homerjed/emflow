from typing import Callable, Literal, Optional
from jaxtyping import PRNGKeyArray, Float, Int, Array, Scalar, jaxtyped
from beartype import beartype as typechecker


typecheck = jaxtyped(typechecker=typechecker)

YArray = Float[Array, "y"] 

YCovariance = Float[Array, "y y"] 

XArray = Float[Array, "x"] 

XCovariance = Float[Array, "x x"]

TCovariance = Float[Array, "x x"] 

OperatorFn = Callable[[XArray], YArray]

OperatorMatrix = Int[Array, "y x"]

XSampleFn = Callable[[PRNGKeyArray], XArray]

XYSampleFn = Callable[[PRNGKeyArray, YArray, Optional[OperatorMatrix]], XArray]
 
SDEType = Literal["non-singular", "zero-ends", "singular", "gamma"]

SampleType = Literal["ddim", "sde", "ode"]

Datasets = Literal["gmm", "moons", "blob", "double-blob", "mnist"]