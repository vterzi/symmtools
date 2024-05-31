"""Quaternion class."""

__all__ = ["Quaternion"]

from numpy import sin, cos, cross

from .const import INF
from .transform import VectorTransformable, Rotation
from .primitive import Point
from .typehints import TypeVar, Any, Float, RealVector

_VectorTransformable = TypeVar(
    "_VectorTransformable", bound=VectorTransformable
)


class Quaternion(VectorTransformable):
    """Quaternion."""

    def __init__(self, scalar: Float, vec: RealVector) -> None:
        """
        Initialize the instance with a scalar `scalar` and a 3D vector `vec`.
        """
        self._scalar = scalar
        super().__init__(vec)

    @property
    def scalar(self) -> Float:
        """Return the scalar part."""
        return self._scalar

    def args(self) -> str:
        return f"{self._scalar},{super().args()}"

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, abs(self._scalar - obj.scalar))
        return res

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self._scalar * other.scalar - self.vec.dot(other.vec),
            cross(self._vec, other.vec)
            + self._scalar * other.vec
            + other.scalar * self._vec,
        )

    def __call__(self, obj: _VectorTransformable) -> _VectorTransformable:
        """Apply the rotation to a transformable object `obj`."""
        res = obj.copy()
        res._vec = (self * Quaternion(0.0, obj.vec) * self.invert()).vec
        return res

    @classmethod
    def from_point(cls, point: Point) -> "Quaternion":
        """Construct an instance from a point `point`."""
        return cls(0.0, point.vec)

    @classmethod
    def from_rotation(cls, rotation: Rotation) -> "Quaternion":
        """Construct an instance from a rotation `rotation`."""
        angle = 0.5 * rotation.angle
        return cls(cos(angle), sin(angle) * rotation.vec)
