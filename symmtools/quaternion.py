"""Quaternion class."""

__all__ = ["Quaternion"]

from math import sin, cos, atan2
from typing import TypeVar, Any

from .const import INF
from .linalg3d import Vector, Matrix, add, mul, lincomb2, dot, norm, cross
from .transform import VectorTransformable, Rotation
from .primitive import Point

_Quaternion = TypeVar("_Quaternion", bound="Quaternion")
_VectorTransformable = TypeVar(
    "_VectorTransformable", bound=VectorTransformable
)


class Quaternion(VectorTransformable):
    """Quaternion."""

    def __init__(self, scalar: float, vec: Vector) -> None:
        """
        Initialize the instance with a scalar `scalar` and a vector `vec`.
        """
        self._scalar = scalar
        super().__init__(vec)

    @property
    def scalar(self) -> float:
        """Scalar part."""
        return self._scalar

    @property
    def args(self) -> str:
        return f"{self._scalar},{super().args}"

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, abs(self._scalar - obj.scalar))
        return res

    def conjugate(self: _Quaternion) -> _Quaternion:
        """Return the conjugate of the instance."""
        return self.invert()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self._scalar * other.scalar - dot(self.vec, other.vec),
            add(
                cross(self._vec, other.vec),
                lincomb2(other.vec, self._scalar, self._vec, other.scalar),
            ),
        )

    def apply(self, vec: Vector) -> Vector:
        """
        Apply the instance to a vector `vec` by multiplying it by the instance
        from the left and by the conjugate of the instance from the right.
        This transformation is a rotation, if the instance is normalized.
        """
        return (self * Quaternion(0.0, vec) * self.conjugate()).vec

    def __call__(self, obj: _VectorTransformable) -> _VectorTransformable:
        """
        Transform the transformable object `obj` by applying the instance to
        its vector.
        """
        res = obj.copy()
        res._vec = self.apply(obj.vec)
        return res

    @property
    def mat(self) -> Matrix:
        """Transformation matrix."""
        vec1 = self.apply((1.0, 0.0, 0.0))
        vec2 = self.apply((0.0, 1.0, 0.0))
        vec3 = self.apply((0.0, 0.0, 1.0))
        return (
            (vec1[0], vec2[0], vec3[0]),
            (vec1[1], vec2[1], vec3[1]),
            (vec1[2], vec2[2], vec3[2]),
        )

    @property
    def rotation(self) -> Rotation:
        """Rotation."""
        vec = self._vec
        vec_norm = norm(vec)
        if vec_norm == 0.0:
            raise ValueError("zero vector as the rotation axis")
        return Rotation(vec, 2.0 * atan2(vec_norm, self._scalar))

    @classmethod
    def from_point(cls, point: Point) -> "Quaternion":
        """Construct an instance from a point `point`."""
        return cls(0.0, point.vec)

    @classmethod
    def from_rotation(cls, rot: Rotation) -> "Quaternion":
        """Construct an instance from a rotation `rot`."""
        angle = 0.5 * rot.angle
        return cls(cos(angle), mul(rot.vec, sin(angle)))
