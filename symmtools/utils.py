"""Utilities."""

__all__ = [
    "clamp",
    "rational",
    "vector",
    "sqnorm",
    "norm",
    "cross",
    "normalize",
    "orthogonalize",
    "canonicalize",
    "diff",
    "same",
    "zero",
    "indep",
    "unitindep",
    "parallel",
    "unitparallel",
    "orthvec",
    "perpendicular",
    "angle",
    "intersectangle",
    "translate",
    "invert",
    "trigrotate",
    "rotate",
    "reflect",
    "trigrotmat",
    "rotmat",
    "reflmat",
    "inertia",
    "signvar",
    "circshift",
]

from math import sqrt, sin, cos, acos
from numpy import array, empty

from .const import PI, PI_2
from .typehints import (
    Sequence,
    Tuple,
    List,
    Bool,
    Int,
    Float,
    Real,
    Vector,
    Matrix,
    RealVector,
    RealVectors,
)

# `max` is faster than `numpy.ndarray.max`
# `float` is faster than `numpy.float64.item`
# `sqrt`, `sin`, `cos`, `acos` from `math` are faster than from `numpy`
# array unpacking is slower than indexing


def clamp(val: Float, low: Float, high: Float) -> Float:
    """Clamp a value `val` within the interval between `low` and `high`."""
    # `numpy.clip` is slower
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


def rational(num: float, tol: float) -> Tuple[int, int]:
    """
    Determine the rational representation (nominator and denominator) of a
    decimal number `num` within a tolerance `tol`.
    """
    negative = num < 0.0
    if negative:
        num = -num
    nom = 0
    denom = 1
    while True:
        diff = num - nom / denom
        if abs(diff) <= tol:
            break
        if diff > 0.0:
            nom += 1
        else:
            denom += 1
    if negative:
        nom = -nom
    return nom, denom


def vector(vec: RealVector) -> Vector:
    """Convert a vector `vec` to a NumPy array of floating-point numbers."""
    return array(vec, dtype=float)


def sqnorm(vec: Vector) -> float:
    """Calculate the squared norm of a vector `vec`."""
    return vec.dot(vec)


def norm(vec: Vector) -> float:
    """Calculate the norm of a vector `vec`."""
    # `numpy.linalg.norm` is slower
    return sqrt(sqnorm(vec))


def cross(vec1: Vector, vec2: Vector) -> Vector:
    """Calculate the cross product of two 3D vectors `vec1` and `vec2`."""
    # `numpy.cross` is slower
    x1 = vec1[0]
    y1 = vec1[1]
    z1 = vec1[2]
    x2 = vec2[0]
    y2 = vec2[1]
    z2 = vec2[2]
    vec = empty(3)
    vec[0] = y1 * z2 - z1 * y2
    vec[1] = z1 * x2 - x1 * z2
    vec[2] = x1 * y2 - y1 * x2
    return vec


def normalize(vec: Vector) -> Vector:
    """Normalize a non-zero vector `vec` to a unit vector."""
    return vec / norm(vec)


def orthogonalize(vec: Vector, unitvec: Vector) -> Vector:
    """Orthogonalize a vector `vec` to a unit vector `unitvec`."""
    return vec - vec.dot(unitvec) * unitvec


def canonicalize(vec: Vector) -> Vector:
    """
    Canonicalize a direction vector `vec` with an undefined sign by making the
    first non-zero component positive.
    """
    for comp in vec:
        if comp < 0.0:
            vec = -vec
        if comp != 0.0:
            break
    return vec


def diff(vec1: Vector, vec2: Vector) -> float:
    """Calculate the difference between two vectors `vec1` and `vec2`."""
    # `norm(vec1 - vec2)` is slower
    return float(max(abs(vec1 - vec2)))


def same(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check whether two vectors `vec1` and `vec2` are the same within a tolerance
    `tol`.
    """
    return diff(vec1, vec2) <= tol


def zero(vec: Vector, tol: float) -> bool:
    """
    Check whether a vector `vec` is a zero vector within a tolerance `tol`.
    """
    return float(max(abs(vec))) <= tol


def indep(vec1: Vector, vec2: Vector) -> float:
    """
    Calculate the linear independence of two 3D vectors `vec1` and `vec2`.
    """
    # `norm(cross(vec1, vec2))` is slower
    return float(max(abs(cross(vec1, vec2))))


def unitindep(unitvec1: Vector, unitvec2: Vector) -> float:
    """
    Calculate the linear independence of two 3D unit vectors `unitvec1` and
    `unitvec2`.
    """
    # `abs(abs(unitvec1.dot(unitvec2)) - 1)` is faster but less accurate
    # `min(diff(unitvec1, unitvec2), diff(unitvec1, -unitvec2))` is slower
    x1 = unitvec1[0]
    y1 = unitvec1[1]
    z1 = unitvec1[2]
    x2 = unitvec2[0]
    y2 = unitvec2[1]
    z2 = unitvec2[2]
    return min(
        max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)),
        max(abs(x1 + x2), abs(y1 + y2), abs(z1 + z2)),
    )


def parallel(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check whether two 3D vectors `vec1` and `vec2` are parallel within a
    tolerance `tol`.
    """
    return indep(vec1, vec2) <= tol


def unitparallel(unitvec1: Vector, unitvec2: Vector, tol: float) -> bool:
    """
    Check whether two 3D unit vectors `unitvec1` and `unitvec2` are parallel
    within a tolerance `tol`.
    """
    return unitindep(unitvec1, unitvec2) <= tol


def perpendicular(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check whether two vectors `vec1` and `vec2` are perpendicular within a
    tolerance `tol`.
    """
    return abs(float(vec1.dot(vec2))) <= tol


def orthvec(unitvec: Vector) -> Vector:
    """
    Generate a unit vector that is orthogonal to a unit vector `unitvec` by
    applying a circular shift to its components, negating the components with
    odd indices, and orthogonalizing the result to `unitvec`.
    """
    n = len(unitvec)
    vec = empty(n)
    change = True
    for i in range(n):
        comp = unitvec[i - 1]
        if change:
            comp = -comp
        vec[i] = comp
        change = not change
    return normalize(orthogonalize(vec, unitvec))


def angle(vec1: Vector, vec2: Vector) -> float:
    """Calculate the angle between two vectors `vec1` and `vec2`."""
    # `acos(clamp(unitvec1.dot(unitvec2), -1.0, 1.0))` is less accurate
    return acos(
        clamp(vec1.dot(vec2) / sqrt(sqnorm(vec1) * sqnorm(vec2)), -1.0, 1.0)
    )


def intersectangle(vec1: Vector, vec2: Vector) -> float:
    """
    Calculate the intersection angle between two lines described by two vectors
    `vec1` and `vec2`.
    """
    ang = angle(vec1, vec2)
    if ang > PI_2:
        ang = PI - ang
    return ang


def translate(vec: Vector, transl: Vector) -> Vector:
    """Translate a vector `vec` by a translation vector `transl`."""
    return vec + transl


def invert(vec: Vector) -> Vector:
    """Invert a vector `vec` through the origin."""
    return -vec


def trigrotate(
    point: Vector, normal: Vector, cos: Float, sin: Float
) -> Vector:
    """
    Rotate a 3D vector `vec` by an angle with cosine `cos` and sine `sin`
    around an axis that contains the origin and is described by a unit vector
    `axis`.
    """
    base = point.dot(normal) * normal
    projection = point - base
    perpendicular = cross(normal, projection)
    return base + projection * cos + perpendicular * sin


def rotate(point: Vector, normal: Vector, angle: Float) -> Vector:
    """
    Rotate a 3D vector `vec` by an angle `angle` around an axis that contains
    the origin and is described by a unit vector `axis`.
    """
    return trigrotate(point, normal, cos(angle), sin(angle))


def reflect(vec: Vector, normal: Vector) -> Vector:
    """
    Reflect a vector `vec` through a plane that contains the origin and whose
    normal is described by a unit vector `normal`.
    """
    return vec - 2.0 * vec.dot(normal) * normal


def trigrotmat(axis: Vector, cos: Float, sin: Float) -> Matrix:
    """
    Generate a 3D transformation matrix for a rotation by an angle with cosine
    `cos` and sine `sin` around an axis that contains the origin and is
    described by a unit vector `axis`.
    """
    x = axis[0]
    y = axis[1]
    z = axis[2]
    xc = x * (1.0 - cos)
    yc = y * (1.0 - cos)
    zc = z * (1.0 - cos)
    xs = x * sin
    ys = y * sin
    zs = z * sin
    xyc = x * yc
    yzc = y * zc
    zxc = z * xc
    mat = empty((3, 3))
    mat[0, 0] = cos + x * xc
    mat[0, 1] = xyc - zs
    mat[0, 2] = zxc + ys
    mat[1, 0] = xyc + zs
    mat[1, 1] = cos + y * yc
    mat[1, 2] = yzc - xs
    mat[2, 0] = zxc - ys
    mat[2, 1] = yzc + xs
    mat[2, 2] = cos + z * zc
    return mat


def rotmat(axis: Vector, angle: Float) -> Matrix:
    """
    Generate a 3D transformation matrix for a rotation by an angle `angle`
    around an axis that contains the origin and is described by a unit vector
    `axis`.
    """
    return trigrotmat(axis, cos(angle), sin(angle))


def reflmat(normal: Vector) -> Matrix:
    """
    Generate a 3D transformation matrix for a reflection through a plane that
    contains the origin and whose normal is described by a unit vector
    `normal`.
    """
    x = normal[0]
    y = normal[1]
    z = normal[2]
    dx = x + x
    dy = y + y
    dz = z + z
    xy = -x * dy
    yz = -y * dz
    zx = -z * dx
    mat = empty((3, 3))
    mat[0, 0] = 1.0 - x * dx
    mat[0, 1] = xy
    mat[0, 2] = zx
    mat[1, 0] = xy
    mat[1, 1] = 1.0 - y * dy
    mat[1, 2] = yz
    mat[2, 0] = zx
    mat[2, 1] = yz
    mat[2, 2] = 1.0 - z * dz
    return mat


def inertia(vecs: Sequence[Vector]) -> Matrix:
    """
    Calculate the inertia tensor of the points of unit mass with 3D positions
    `vecs`.
    """
    xx = 0.0
    yy = 0.0
    zz = 0.0
    xy = 0.0
    zx = 0.0
    yz = 0.0
    for vec in vecs:
        x = vec[0]
        y = vec[1]
        z = vec[2]
        xs = x * x
        ys = y * y
        zs = z * z
        xx += ys + zs
        yy += zs + xs
        zz += xs + ys
        xy -= x * y
        zx -= x * z
        yz -= y * z
    mat = empty((3, 3))
    mat[0, 0] = xx
    mat[0, 1] = xy
    mat[0, 2] = zx
    mat[1, 0] = xy
    mat[1, 1] = yy
    mat[1, 2] = yz
    mat[2, 0] = zx
    mat[2, 1] = yz
    mat[2, 2] = zz
    return mat


def signvar(
    vec: RealVector, parity: Int = 0, indep: Bool = False
) -> List[List[Real]]:
    """
    Generate vectors with all possible sign changes of the components of a
    vector `vec` that satisfy a parity `parity`.  If `parity` is
    positive/negative, only the vectors resulting from even/odd number of sign
    changes are returned.  If `parity` is zero, all vectors are returned.  If
    `indep` is enabled, only linearly independent vectors are returned.
    """
    nonzeros = 0
    for comp in vec:
        if comp != 0:
            nonzeros += 1
    arr = []
    excluded = []
    for n in range(2**nonzeros):
        changes = nonzeros * [False]
        i = 0
        sign = 1
        while n > 0:
            if n % 2 == 1:
                changes[i] = True
                sign = -sign
            i += 1
            n //= 2
        if sign * parity >= 0:
            if indep:
                if changes in excluded:
                    continue
                excluded.append([not change for change in changes])
            elem = []
            i = 0
            for comp in vec:
                if comp != 0:
                    if changes[i]:
                        comp = -comp
                    i += 1
                elem.append(comp)
            arr.append(elem)
    return arr


def circshift(vecs: RealVectors) -> List[List[Real]]:
    """
    Generate all possible vectors by applying circular shifts on the
    components of vectors `vecs`.
    """
    arr = []
    for vec in vecs:
        n = len(vec)
        for i in range(n):
            elem = []
            for ii in range(n, 0, -1):
                elem.append(vec[i - ii])
            arr.append(elem)
    return arr
