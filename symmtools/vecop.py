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
    "parallelunit",
    "perpendicular",
    "translate",
    "invert",
    "move2",
    "rotate",
    "vecrotate",
    "reflect",
    "rotmat",
    "reflmat",
]

from numpy import array, sin, cos, cross
from numpy.linalg import norm

from .typehints import Float, Vector, Matrix, RealVector

# `max` is faster than `numpy.ndarray.max`
# `float` is faster than `numpy.float64.item`


def vector(vec: RealVector) -> Vector:
    """Convert a vector `vec` to a NumPy array of floating-point numbers."""
    return array(vec, dtype=float)


def canonicalize(vec: Vector) -> Vector:
    """
    Canonicalize an unsigned direction vector `vec` by making the first
    non-zero coordinate positive.
    """
    for coord in vec:
        if coord < 0.0:
            vec = -vec
        if coord != 0.0:
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
    # `norm(vec1 - vec2)` is slower
    return float(max(abs(vec1 - vec2)))


def same(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two vectors `vec1` and `vec2` are the same within a tolerance
    `tol`.
    """
    return diff(vec1, vec2) <= tol


def indep(vec1: Vector, vec2: Vector) -> float:
    """Calculate the linear independence of two vectors `vec1` and `vec2`."""
    # `norm(cross(vec1, vec2))` is slower
    return float(max(abs(cross(vec1, vec2))))


def indepunit(vec1: Vector, vec2: Vector) -> float:
    """
    Calculate the linear independence of two unit vectors `vec1` and `vec2`.
    """
    # `abs(abs(vec1.dot(vec2)) - 1)` is faster but less accurate
    return min(diff(vec1, vec2), diff(vec1, -vec2))


def parallel(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two vectors `vec1` and `vec2` are parallel within a tolerance
    `tol`.
    """
    return indep(vec1, vec2) <= tol


def parallelunit(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two unit vectors `vec1` and `vec2` are parallel within a
    tolerance `tol`.
    """
    return indepunit(vec1, vec2) <= tol


def perpendicular(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check wether two vectors `vec1` and `vec2` are perpendicular within a
    tolerance `tol`.
    """
    return abs(float(vec1.dot(vec2))) <= tol


def translate(vec: Vector, translation: Vector) -> Vector:
    """Translate a vector `vec` by a translation vector `translation`."""
    return vec + translation


def invert(vec: Vector) -> Vector:
    """Invert a vector `vec` through the origin."""
    return -vec


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


def rotate(vec: Vector, rotation: Vector, angle: Float) -> Vector:
    """
    Rotate a 3D vector `vec` by an angle `angle` around an axis that contains
    the origin and is described by a unit vector `rotation`.
    """
    return move2(vec, rotation, cos(angle), sin(angle))


def vecrotate(vec: Vector, rotation: Vector) -> Vector:
    """Rotate a 3D vector `vec` by a rotation vector `rotation`."""
    length = norm(rotation)
    if length > 0.0:
        vec = rotate(vec, rotation / length, length)
    return vec


def reflect(vec: Vector, reflection: Vector) -> Vector:
    """
    Reflect a vector `vec` through a plane that contains the origin and whose
    normal is described by a unit vector `reflection`.
    """
    return vec - 2.0 * vec.dot(reflection) * reflection


def rotmat(rotation: Vector, angle: Float) -> Matrix:
    """
    Generate a 3D transformation matrix for a rotation by an angle `angle`
    around an axis that contains the origin and is described by a unit vector
    `rotation`.
    """
    x, y, z = rotation
    c = cos(angle)
    s = sin(angle)
    xc = x * (1.0 - c)
    yc = y * (1.0 - c)
    zc = z * (1.0 - c)
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


def reflmat(reflection: Vector) -> Matrix:
    """
    Generate a 3D transformation matrix for a reflection through a plane that
    contains the origin and whose normal is described by a unit vector
    `reflection`.
    """
    x, y, z = reflection
    x_ = x + x
    y_ = y + y
    z_ = z + z
    xy = -x * y_
    yz = -y * z_
    zx = -z * x_
    return array(
        [
            [1.0 - x * x_, xy, zx],
            [xy, 1.0 - y * y_, yz],
            [zx, yz, 1.0 - z * z_],
        ]
    )
