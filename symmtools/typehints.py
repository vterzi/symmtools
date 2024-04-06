"""Type hints."""

__all__ = [
    "Bool",
    "Int",
    "Float",
    "Complex",
    "Scalar",
    "Vector",
    "Matrix",
]

from typing import Union

from numpy import bool_, signedinteger, floating, complexfloating
from numpy.typing import NDArray

Bool = Union[bool, bool_]
Int = Union[int, signedinteger]
Float = Union[float, floating]
Complex = Union[complex, complexfloating]
Scalar = Union[Int, Float, Complex]
Vector = NDArray[floating]
Matrix = NDArray[floating]
