"""Functions for basic vector operations."""

__all__ = [
    "vector",
    "canonicalize",
    "normalize",
    "orthogonalize",
    "diff",
    "same",
    "indep",
    "indepunit",
    "parallel",
    "perpendicular",
    "translate",
    "invert",
    "move2",
    "rotate",
    "rotate_",
    "reflect",
    "reflect_",
    "rotmat",
    "reflmat",
]

from numpy import array, sin, cos, cross
from numpy.linalg import norm

from .typehints import Float, Vector, Matrix, RealVector


def vector(vec: RealVector) -> Vector:
    """Convert a vector `vec` to a NumPy array of floating-point numbers."""
    return array(vec, dtype=float)


def canonicalize(vec: Vector) -> Vector:
    """
    Canonicalize an unsigned direction vector `vec` by making the first
    non-zero coordinate positive.
    """
    for coord in vec:
        if coord < 0:
            vec = -vec
        if coord != 0:
            break
    return vec


def normalize(vec: Vector) -> Vector:
    """Normalize a non-zero vector `vec` to a unit vector."""
    return vec / norm(vec)


def orthogonalize(vec1: Vector, vec2: Vector) -> Vector:
    """Orthogonalize a vector `vec1` to a unit vector `vec2`."""
    return vec1 - vec1.dot(vec2) * vec2


def diff(vec1: Vector, vec2: Vector) -> float:
    """Calculate the difference between two vectors `vec1` and `vec2`."""
    return abs(vec1 - vec2).max().item()  # norm(vec1 - vec2)


def same(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two vectors `vec1` and `vec2` are the same within a tolerance
    `tol`.
    """
    return diff(vec1, vec2) <= tol


def indep(vec1: Vector, vec2: Vector) -> float:
    """Calculate the linear independence of two vectors `vec1` and `vec2`."""
    return abs(cross(vec1, vec2)).max().item()  # norm(cross(vec1, vec2))


def indepunit(vec1: Vector, vec2: Vector) -> float:
    """
    Calculate the linear independence of two unit vectors `vec1` and `vec2`.
    """
    return min(diff(vec1, vec2), diff(vec1, -vec2))


def parallel(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two vectors `vec1` and `vec2` are parallel within a tolerance
    `tol`.
    """
    return indep(vec1, vec2) <= tol


def perpendicular(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two vectors `vec1` and `vec2` are perpendicular within a
    tolerance `tol`.
    """
    return abs(vec1.dot(vec2).item()) <= tol


def translate(point: Vector, translation: Vector) -> Vector:
    """Translate a point `point` by a translation vector `translation`."""
    return point + translation


def invert(point: Vector) -> Vector:
    """Invert a point `point` through the origin."""
    return -point


def move2(point: Vector, normal: Vector, coef1: Float, coef2: Float) -> Vector:
    """
    Move a 3D point `point` in a plane containing the point with a normalized
    normal `normal` to the position represented by the linear combination of
    the projection of the point position on the plane scaled by `coef1` and its
    perpendicular in the plane scaled by `coef2`.
    """
    base = point.dot(normal) * normal
    projection = point - base
    perpendicular = cross(normal, projection)
    return base + projection * coef1 + perpendicular * coef2


def rotate(point: Vector, rotation: Vector, angle: Float) -> Vector:
    """
    Rotate a 3D point `point` around a rotation axis with a normalized
    direction `rotation` by an angle `angle`.
    """
    return move2(point, rotation, cos(angle), sin(angle))


def rotate_(point: Vector, rotation: Vector) -> Vector:
    """Rotate a 3D point `point` by a rotation vector `rotation`."""
    length = norm(rotation)
    if length > 0:
        point = rotate(point, rotation / length, length)
    return point


def reflect(point: Vector, reflection: Vector) -> Vector:
    """
    Reflect a point `point` through a reflection plane with a normalized normal
    `reflection`.
    """
    return point - 2 * point.dot(reflection) * reflection


def reflect_(point: Vector, reflection: Vector) -> Vector:
    """
    Reflect a point `point` through a reflection plane with a normal
    `reflection`.
    """
    length = norm(reflection)
    if length > 0:
        point = reflect(point, reflection / length)
    return point


def rotmat(vec: Vector, angle: Float) -> Matrix:
    """
    Return the transformation matrix for a rotation by an angle `angle` around
    an axis that contains the origin and is described by a unit vector `vec`.
    """
    x, y, z = vec
    c = cos(angle)
    s = sin(angle)
    xc = x * (1 - c)
    yc = y * (1 - c)
    zc = z * (1 - c)
    xs = x * s
    ys = y * s
    zs = z * s
    xyc = x * yc
    yzc = y * zc
    zxc = z * xc
    return array(
        [
            [c + x * xc, xyc - zs, zxc + ys],
            [xyc + zs, c + y * yc, yzc - xs],
            [zxc - ys, yzc + xs, c + z * zc],
        ]
    )


def reflmat(vec: Vector) -> Matrix:
    """
    Return the transformation matrix for a reflection through a plane that
    contains the origin and whose normal is described by a unit vector `vec`.
    """
    x, y, z = vec
    x_ = x + x
    y_ = y + y
    z_ = z + z
    xy = -x * y_
    yz = -y * z_
    zx = -z * x_
    return array(
        [
            [1 - x * x_, xy, zx],
            [xy, 1 - y * y_, yz],
            [zx, yz, 1 - z * z_],
        ]
    )
