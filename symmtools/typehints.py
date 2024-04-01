"""Type hints."""

__all__ = [
    "Bool",
    "Int",
    "Float",
    "Complex",
    "Vector",
]

from typing import TypeVar

from numpy import bool_, signedinteger, floating, complexfloating
from numpy.typing import NDArray

Bool = TypeVar("Bool", bool, bool_)
Int = TypeVar("Int", int, signedinteger)
Float = TypeVar("Float", float, floating)
Complex = TypeVar("Complex", complex, complexfloating)
Vector = NDArray[floating]
