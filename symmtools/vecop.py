"""Functions for basic vector operations."""

__all__ = (
    "Vector",
    "vector",
    "normalize",
    "same",
    "parallel",
    "perpendicular",
    "translate",
    "invert",
    "move2",
    "rotate",
    "reflect",
)

from collections.abc import Sequence
from typing import TypeVar, Union

from numpy import (
    array,
    sin,
    cos,
    dot,
    cross,
    bool_,
    signedinteger,
    floating,
    complexfloating,
)
from numpy.linalg import norm
from numpy.typing import NDArray

Bool = TypeVar("Bool", bool, bool_)
Int = TypeVar("Int", int, signedinteger)
Float = TypeVar("Float", float, floating)
Complex = TypeVar("Complex", complex, complexfloating)
Vector = NDArray[floating]


def vector(vec: Sequence[Union[Int, Float]]) -> Vector:
    """Convert a vector `vec` to a NumPy array of floating-point numbers."""
    return array(vec, dtype=float)


def normalize(vec: Vector) -> Vector:
    """Normalize a vector `vec` to a unit vector."""
    return vec / norm(vec)


def same(vec1: Vector, vec2: Vector, tol: Float) -> bool_:
    """
    Check if two vectors `vec1` and `vec2` are the same within a tolerance
    `tol`.
    """
    return norm(vec1 - vec2) <= tol


def parallel(vec1: Vector, vec2: Vector, tol: Float) -> bool_:
    """
    Check if two vectors `vec1` and `vec2` are parallel within a tolerance
    `tol`.
    """
    return norm(cross(vec1, vec2)) <= tol


def perpendicular(vec1: Vector, vec2: Vector, tol: Float) -> bool_:
    """
    Check if two vectors `vec1` and `vec2` are perpendicular within a tolerance
    `tol`.
    """
    return abs(dot(vec1, vec2)) <= tol


def translate(point: Vector, translation: Vector) -> Vector:
    """Translate a point `point` by a translation vector `translation`."""
    return point + translation


def invert(point: Vector) -> Vector:
    """Invert a point `point` through the origin."""
    return -point


def move2(point: Vector, normal: Vector, coef1: Float, coef2: Float) -> Vector:
    """
    Move a point `point` in a plane containing the point with a normalized
    normal `normal` to a position represented by the linear combination of the
    projection of the point position on the plane scaled by `coef1` and its
    perpendicular in the plane scaled by `coef2`.
    """
    base = dot(point, normal) * normal
    projection = point - base
    perpendicular_ = cross(normal, projection)
    return base + projection * coef1 + perpendicular_ * coef2


def rotate(point: Vector, rotation: Vector, angle: Float) -> Vector:
    """
    Rotate a point `point` around a rotation axis with a normalized direction
    `rotation` by an angle `angle`.
    """
    return move2(point, rotation, cos(angle), sin(angle))


def reflect(point: Vector, reflection: Vector) -> Vector:
    """
    Reflect a point `point` through a reflection plane with a normalized normal
    `reflection`.
    """
    return point - 2 * dot(point, reflection) * reflection
