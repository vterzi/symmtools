"""Type hints."""

__all__ = [
    "Union",
    "Optional",
    "Any",
    "Sequence",
    "Tuple",
    "List",
    "Dict",
    "Bool",
    "Int",
    "Float",
    "Complex",
    "Real",
    "Scalar",
    "Bools",
    "Ints",
    "Floats",
    "Complexes",
    "Reals",
    "Scalars",
    "Vector",
    "Matrix",
    "RealVector",
    "RealVectors",
]

from typing import Union, Optional, Any, Sequence, Tuple, List, Dict

from numpy import bool_, signedinteger, floating, complexfloating
from numpy.typing import NDArray

Bool = Union[bool, bool_]
Int = Union[int, signedinteger]
Float = Union[float, floating]
Complex = Union[complex, complexfloating]
Real = Union[Int, Float]
Scalar = Union[Int, Float, Complex]
Bools = Sequence[Bool]
Ints = Sequence[Int]
Floats = Sequence[Float]
Complexes = Sequence[Complex]
Reals = Sequence[Real]
Scalars = Sequence[Scalar]
Vector = NDArray[floating]
Matrix = NDArray[floating]
RealVector = Union[Vector, Reals]
RealVectors = Sequence[RealVector]
