"""Linear algebra functions for 3D vectors and matrices."""

__all__ = [
    "Vector",
    "Matrix",
    "Quaternion",
    "vector",
    "matrix",
    "pos",
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "lincomb2",
    "lincomb3",
    "dot",
    "sqnorm",
    "norm",
    "cross",
    "normalize",
    "orthogonalize",
    "canonicalize",
    "zero",
    "diff",
    "same",
    "indep",
    "unitindep",
    "parallel",
    "unitparallel",
    "perpendicular",
    "orthvec",
    "angle",
    "intersectangle",
    "normangle",
    "transpose",
    "matmulvec",
    "vecmulmat",
    "matmulmat",
    "trigrotate",
    "rotate",
    "reflect",
    "trigrotmat",
    "rotmat",
    "reflmat",
    "rotquat",
    "quatrot",
    "conjugate",
    "quatmulquat",
    "quatrotate",
    "quatzyzangles",
    "alignvec",
    "alignvecs",
    "inertia",
    "symmeig",
]

from math import (
    copysign,
    fmod,
    gcd,
    factorial,
    comb,
    sqrt,
    hypot,
    cos,
    sin,
    acos,
    atan2,
)
from cmath import exp
from typing import Optional, Sequence, Tuple, List, Dict

try:
    from numpy.linalg import eigh

    EIGH_AVAIL = True
except ImportError:
    EIGH_AVAIL = False

from .const import PI, TAU, HALF_PI, EPS
from .utils import clamp, reducefrac, sqrtfactor

# `numpy` with `numpy.ndarray` is slower than `math` with `tuple`.
# Array unpacking is slower than indexing.

Vector = Tuple[float, float, float]
Matrix = Tuple[Vector, Vector, Vector]
Quaternion = Tuple[float, float, float, float]

_SQRT_HALF = sqrt(0.5)


def vector(arr: Sequence[float]) -> Vector:
    """Create a vector from an array `arr`."""
    return (arr[0], arr[1], arr[2])


def matrix(arr: Sequence[Sequence[float]]) -> Matrix:
    """Create a matrix from an array `arr`."""
    return (vector(arr[0]), vector(arr[1]), vector(arr[2]))


def pos(vec: Vector) -> Vector:
    """Return the positive of a vector `vec`."""
    return vec


def neg(vec: Vector) -> Vector:
    """Return the negative of a vector `vec`."""
    return (-vec[0], -vec[1], -vec[2])


def add(vec1: Vector, vec2: Vector) -> Vector:
    """Add two vectors `vec1` and `vec2`."""
    return (vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2])


def sub(vec1: Vector, vec2: Vector) -> Vector:
    """Subtract two vectors `vec1` and `vec2`."""
    return (vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2])


def mul(vec: Vector, scalar: float) -> Vector:
    """Multiply a vector `vec` by a scalar `scalar`."""
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def div(vec: Vector, scalar: float) -> Vector:
    """Divide a vector `vec` by a scalar `scalar`."""
    return mul(vec, 1.0 / scalar)


def lincomb2(
    vec1: Vector, scalar1: float, vec2: Vector, scalar2: float
) -> Vector:
    """
    Calculate the linear combination of two vectors `vec1` and `vec2` with
    scalars `scalar1` and `scalar2`.
    """
    return (
        vec1[0] * scalar1 + vec2[0] * scalar2,
        vec1[1] * scalar1 + vec2[1] * scalar2,
        vec1[2] * scalar1 + vec2[2] * scalar2,
    )


def lincomb3(
    vec1: Vector,
    scalar1: float,
    vec2: Vector,
    scalar2: float,
    vec3: Vector,
    scalar3: float,
) -> Vector:
    """
    Calculate the linear combination of three vectors `vec1`, `vec2`, and
    `vec3` with scalars `scalar1`, `scalar2`, and `scalar3`.
    """
    return (
        vec1[0] * scalar1 + vec2[0] * scalar2 + vec3[0] * scalar3,
        vec1[1] * scalar1 + vec2[1] * scalar2 + vec3[1] * scalar3,
        vec1[2] * scalar1 + vec2[2] * scalar2 + vec3[2] * scalar3,
    )


def dot(vec1: Vector, vec2: Vector) -> float:
    """Calculate the dot product of two vectors `vec1` and `vec2`."""
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def sqnorm(vec: Vector) -> float:
    """Calculate the squared norm of a vector `vec`."""
    x = vec[0]
    y = vec[1]
    z = vec[2]
    return x * x + y * y + z * z


def norm(vec: Vector) -> float:
    """Calculate the norm of a vector `vec`."""
    return hypot(vec[0], vec[1], vec[2])


def cross(vec1: Vector, vec2: Vector) -> Vector:
    """Calculate the cross product of two vectors `vec1` and `vec2`."""
    x1 = vec1[0]
    y1 = vec1[1]
    z1 = vec1[2]
    x2 = vec2[0]
    y2 = vec2[1]
    z2 = vec2[2]
    return (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)


def normalize(vec: Vector) -> Vector:
    """Normalize a non-zero vector `vec` to a unit vector."""
    x = vec[0]
    y = vec[1]
    z = vec[2]
    scalar = 1.0 / hypot(x, y, z)
    return (x * scalar, y * scalar, z * scalar)


def orthogonalize(vec: Vector, unitvec: Vector) -> Vector:
    """Orthogonalize a vector `vec` to a unit vector `unitvec`."""
    # `sub(vec, mul(unitvec, dot(vec, unitvec)))` is slower.
    x1 = vec[0]
    y1 = vec[1]
    z1 = vec[2]
    x2 = unitvec[0]
    y2 = unitvec[1]
    z2 = unitvec[2]
    scalar = x1 * x2 + y1 * y2 + z1 * z2
    return (x1 - scalar * x2, y1 - scalar * y2, z1 - scalar * z2)


def canonicalize(vec: Vector) -> Vector:
    """
    Canonicalize a direction vector `vec` with an undefined sign by making the
    first non-zero component positive.
    """
    for comp in vec:
        if comp < 0.0:
            vec = neg(vec)
        if comp != 0.0:
            break
    return vec


def zero(vec: Vector, tol: float) -> bool:
    """
    Check whether a vector `vec` is a zero vector within a tolerance `tol`.
    """
    return max(abs(vec[0]), abs(vec[1]), abs(vec[2])) <= tol


def diff(vec1: Vector, vec2: Vector) -> float:
    """Calculate the difference between two vectors `vec1` and `vec2`."""
    # `norm(sub(vec1, vec2))` is slower.
    return max(
        abs(vec1[0] - vec2[0]), abs(vec1[1] - vec2[1]), abs(vec1[2] - vec2[2])
    )


def same(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check whether two vectors `vec1` and `vec2` are the same within a tolerance
    `tol`.
    """
    return diff(vec1, vec2) <= tol


def indep(vec1: Vector, vec2: Vector) -> float:
    """Calculate the linear independence of two vectors `vec1` and `vec2`."""
    # `norm(cross(vec1, vec2))` is slower.
    x1 = vec1[0]
    y1 = vec1[1]
    z1 = vec1[2]
    x2 = vec2[0]
    y2 = vec2[1]
    z2 = vec2[2]
    return max(
        abs(y1 * z2 - z1 * y2), abs(z1 * x2 - x1 * z2), abs(x1 * y2 - y1 * x2)
    )


def unitindep(unitvec1: Vector, unitvec2: Vector) -> float:
    """
    Calculate the linear independence of two unit vectors `unitvec1` and
    `unitvec2`.
    """
    # `abs(abs(dot(unitvec1, unitvec2)) - 1)` is faster but less accurate.
    # `min(diff(unitvec1, unitvec2), diff(unitvec1, neg(unitvec2)))` is slower.
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
    Check whether two vectors `vec1` and `vec2` are parallel within a tolerance
    `tol`.
    """
    return indep(vec1, vec2) <= tol


def unitparallel(unitvec1: Vector, unitvec2: Vector, tol: float) -> bool:
    """
    Check whether two unit vectors `unitvec1` and `unitvec2` are parallel
    within a tolerance `tol`.
    """
    return unitindep(unitvec1, unitvec2) <= tol


def perpendicular(vec1: Vector, vec2: Vector, tol: float) -> bool:
    """
    Check whether two vectors `vec1` and `vec2` are perpendicular within a
    tolerance `tol`.
    """
    return abs(dot(vec1, vec2)) <= tol


def orthvec(unitvec: Vector) -> Vector:
    """
    Create a unit vector that is orthogonal to a unit vector `unitvec` by
    applying a circular shift to its components, negating the components with
    odd indices, and orthogonalizing the result to the original unit vector.
    """
    # `normalize(orthogonalize((-unitvec[2], unitvec[0], -unitvec[1]),
    # unitvec))` is slower.
    x2 = unitvec[0]
    y2 = unitvec[1]
    z2 = unitvec[2]
    x1 = -z2
    y1 = x2
    z1 = -y2
    scalar = x1 * x2 + y1 * y2 + z1 * z2
    x1 -= scalar * x2
    y1 -= scalar * y2
    z1 -= scalar * z2
    scalar = 1.0 / hypot(x1, y1, z1)
    return (x1 * scalar, y1 * scalar, z1 * scalar)


def angle(vec1: Vector, vec2: Vector) -> float:
    """Calculate the angle between two vectors `vec1` and `vec2`."""
    # `acos(clamp(dot(unitvec1, unitvec2), -1.0, 1.0))` is less accurate.
    # `acos(clamp(dot(vec1, vec2) / sqrt(sqnorm(vec1) * sqnorm(vec2)), -1.0,
    # 1.0))` is slower.
    x1 = vec1[0]
    y1 = vec1[1]
    z1 = vec1[2]
    x2 = vec2[0]
    y2 = vec2[1]
    z2 = vec2[2]
    return acos(
        clamp(
            (x1 * x2 + y1 * y2 + z1 * z2)
            / sqrt(
                (x1 * x1 + y1 * y1 + z1 * z1) * (x2 * x2 + y2 * y2 + z2 * z2)
            ),
            -1.0,
            1.0,
        )
    )


def intersectangle(vec1: Vector, vec2: Vector) -> float:
    """
    Calculate the intersection angle between two lines described by two vectors
    `vec1` and `vec2`.
    """
    ang = angle(vec1, vec2)
    if ang > HALF_PI:
        ang = PI - ang
    return ang


def normangle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi)."""
    return fmod(angle + PI, TAU) - PI


def transpose(mat: Matrix) -> Matrix:
    """Transpose a matrix `mat`."""
    row0 = mat[0]
    row1 = mat[1]
    row2 = mat[2]
    return (
        (row0[0], row1[0], row2[0]),
        (row0[1], row1[1], row2[1]),
        (row0[2], row1[2], row2[2]),
    )


def matmulvec(mat: Matrix, vec: Vector) -> Vector:
    """Multiply a matrix `mat` by a vector `vec`."""
    row0 = mat[0]
    row1 = mat[1]
    row2 = mat[2]
    x = vec[0]
    y = vec[1]
    z = vec[2]
    return (
        row0[0] * x + row0[1] * y + row0[2] * z,
        row1[0] * x + row1[1] * y + row1[2] * z,
        row2[0] * x + row2[1] * y + row2[2] * z,
    )


def vecmulmat(vec: Vector, mat: Matrix) -> Vector:
    """Multiply a vector `vec` by a matrix `mat`."""
    x = vec[0]
    y = vec[1]
    z = vec[2]
    row0 = mat[0]
    row1 = mat[1]
    row2 = mat[2]
    return (
        x * row0[0] + y * row1[0] + z * row2[0],
        x * row0[1] + y * row1[1] + z * row2[1],
        x * row0[2] + y * row1[2] + z * row2[2],
    )


def matmulmat(mat1: Matrix, mat2: Matrix) -> Matrix:
    """Multiply two matrices `mat1` and `mat2`."""
    row0 = mat1[0]
    row1 = mat1[1]
    row2 = mat1[2]
    xx1 = row0[0]
    xy1 = row0[1]
    xz1 = row0[2]
    yx1 = row1[0]
    yy1 = row1[1]
    yz1 = row1[2]
    zx1 = row2[0]
    zy1 = row2[1]
    zz1 = row2[2]
    row0 = mat2[0]
    row1 = mat2[1]
    row2 = mat2[2]
    xx2 = row0[0]
    xy2 = row0[1]
    xz2 = row0[2]
    yx2 = row1[0]
    yy2 = row1[1]
    yz2 = row1[2]
    zx2 = row2[0]
    zy2 = row2[1]
    zz2 = row2[2]
    return (
        (
            xx1 * xx2 + xy1 * yx2 + xz1 * zx2,
            xx1 * xy2 + xy1 * yy2 + xz1 * zy2,
            xx1 * xz2 + xy1 * yz2 + xz1 * zz2,
        ),
        (
            yx1 * xx2 + yy1 * yx2 + yz1 * zx2,
            yx1 * xy2 + yy1 * yy2 + yz1 * zy2,
            yx1 * xz2 + yy1 * yz2 + yz1 * zz2,
        ),
        (
            zx1 * xx2 + zy1 * yx2 + zz1 * zx2,
            zx1 * xy2 + zy1 * yy2 + zz1 * zy2,
            zx1 * xz2 + zy1 * yz2 + zz1 * zz2,
        ),
    )


def trigrotate(vec: Vector, axis: Vector, cos: float, sin: float) -> Vector:
    """
    Rotate a vector `vec` by an angle with cosine `cos` and sine `sin` around
    an axis that contains the origin and is described by a unit vector `axis`.
    """
    # Rodrigues' rotation formula.
    # `lincomb3(vec, cos, cross(axis, vec), sin, axis, dot(axis, vec)
    # * (1.0 - cos))` is slower.
    vec_x = vec[0]
    vec_y = vec[1]
    vec_z = vec[2]
    axis_x = axis[0]
    axis_y = axis[1]
    axis_z = axis[2]
    perpendicular_x = axis_y * vec_z - axis_z * vec_y
    perpendicular_y = axis_z * vec_x - axis_x * vec_z
    perpendicular_z = axis_x * vec_y - axis_y * vec_x
    scalar = (axis_x * vec_x + axis_y * vec_y + axis_z * vec_z) * (1.0 - cos)
    return (
        vec_x * cos + perpendicular_x * sin + axis_x * scalar,
        vec_y * cos + perpendicular_y * sin + axis_y * scalar,
        vec_z * cos + perpendicular_z * sin + axis_z * scalar,
    )


def rotate(vec: Vector, axis: Vector, angle: float) -> Vector:
    """
    Rotate a vector `vec` by an angle `angle` around an axis that contains the
    origin and is described by a unit vector `axis`.
    """
    return trigrotate(vec, axis, cos(angle), sin(angle))


def reflect(vec: Vector, normal: Vector) -> Vector:
    """
    Reflect a vector `vec` through a plane that contains the origin and whose
    normal is described by a unit vector `normal`.
    """
    # `sub(vec, mul(normal, 2.0 * dot(vec, normal)))` is slower.
    vec_x = vec[0]
    vec_y = vec[1]
    vec_z = vec[2]
    normal_x = normal[0]
    normal_y = normal[1]
    normal_z = normal[2]
    scalar = vec_x * normal_x + vec_y * normal_y + vec_z * normal_z
    scalar += scalar
    return (
        vec_x - normal_x * scalar,
        vec_y - normal_y * scalar,
        vec_z - normal_z * scalar,
    )


def trigrotmat(axis: Vector, cos: float, sin: float) -> Matrix:
    """
    Create a transformation matrix for a rotation by an angle with cosine
    `cos` and sine `sin` around an axis that contains the origin and is
    described by a unit vector `axis`.
    """
    x = axis[0]
    y = axis[1]
    z = axis[2]
    temp = 1.0 - cos
    x_cos = x * temp
    y_cos = y * temp
    z_cos = z * temp
    x_sin = x * sin
    y_sin = y * sin
    z_sin = z * sin
    x_y_cos = x * y_cos
    y_z_cos = y * z_cos
    z_x_cos = z * x_cos
    return (
        (cos + x * x_cos, x_y_cos - z_sin, z_x_cos + y_sin),
        (x_y_cos + z_sin, cos + y * y_cos, y_z_cos - x_sin),
        (z_x_cos - y_sin, y_z_cos + x_sin, cos + z * z_cos),
    )


def rotmat(axis: Vector, angle: float) -> Matrix:
    """
    Create a transformation matrix for a rotation by an angle `angle` around
    an axis that contains the origin and is described by a unit vector `axis`.
    """
    return trigrotmat(axis, cos(angle), sin(angle))


def reflmat(normal: Vector) -> Matrix:
    """
    Create a transformation matrix for a reflection through a plane that
    contains the origin and whose normal is described by a unit vector
    `normal`.
    """
    x = normal[0]
    y = normal[1]
    z = normal[2]
    x2 = x + x
    y2 = y + y
    z2 = z + z
    xy = -x * y2
    yz = -y * z2
    zx = -z * x2
    return (
        (1.0 - x * x2, xy, zx),
        (xy, 1.0 - y * y2, yz),
        (zx, yz, 1.0 - z * z2),
    )


def rotquat(axis: Vector, angle: float) -> Quaternion:
    """
    Create a rotation quaternion from a unit vector `axis` representing the
    axis of rotation and an angle `angle`.
    """
    angle *= 0.5
    scalar = sin(angle)
    return (cos(angle), axis[0] * scalar, axis[1] * scalar, axis[2] * scalar)


def quatrot(quat: Quaternion) -> Tuple[Vector, float]:
    """Determine the axis and angle of a rotation quaternion `quat`."""
    x = quat[1]
    y = quat[2]
    z = quat[3]
    norm = hypot(x, y, z)
    scalar = 1.0 / norm
    return (x * scalar, y * scalar, z * scalar), 2.0 * atan2(norm, quat[0])


def conjugate(quat: Quaternion) -> Quaternion:
    """Return the conjugate of a quaternion `quat`."""
    return (quat[0], -quat[1], -quat[2], -quat[3])


def quatmulquat(quat1: Quaternion, quat2: Quaternion) -> Quaternion:
    """Multiply two quaternions `quat1` and `quat2`."""
    w1 = quat1[0]
    x1 = quat1[1]
    y1 = quat1[2]
    z1 = quat1[3]
    w2 = quat2[0]
    x2 = quat2[1]
    y2 = quat2[2]
    z2 = quat2[3]
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        y1 * z2 - y2 * z1 + x1 * w2 + x2 * w1,
        z1 * x2 - z2 * x1 + y1 * w2 + y2 * w1,
        x1 * y2 - x2 * y1 + z1 * w2 + z2 * w1,
    )


def quatrotate(vec: Vector, quat: Quaternion) -> Vector:
    """Apply a rotation quaternion `quat` to a vector `vec`."""
    vec_x = vec[0]
    vec_y = vec[1]
    vec_z = vec[2]
    quat_w = quat[0]
    quat_x = quat[1]
    quat_y = quat[2]
    quat_z = quat[3]
    temp_x = quat_y * vec_z - quat_z * vec_y
    temp_y = quat_z * vec_x - quat_x * vec_z
    temp_z = quat_x * vec_y - quat_y * vec_x
    temp_x += temp_x
    temp_y += temp_y
    temp_z += temp_z
    return (
        vec_x + quat_w * temp_x + quat_y * temp_z - quat_z * temp_y,
        vec_y + quat_w * temp_y + quat_z * temp_x - quat_x * temp_z,
        vec_z + quat_w * temp_z + quat_x * temp_y - quat_y * temp_x,
    )


def quatzyzangles(quat: Quaternion) -> Tuple[float, float, float]:
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    angle2 = atan2(hypot(x, y), hypot(w, z))  # check gimbal lock
    angle2 += angle2
    angle_plus = atan2(z, w)
    angle_minus = atan2(x, y)
    angle1 = normangle(angle_plus + angle_minus)
    angle3 = normangle(angle_plus - angle_minus)
    return (angle1, angle2, angle3)


def alignvec(
    from_vec: Vector, to_vec: Vector, orth_axis: Optional[Vector] = None
) -> Quaternion:
    """
    Return the proper transformation from one non-zero vector `from_vec` to
    another non-zero vector `to_vec`.  If a vector `orth_axis` that is
    orthogonal to vector `to_vec` is provided, it will be used to construct
    the rotation in case the vectors are antiparallel.
    """
    axis = cross(from_vec, to_vec)
    if sqnorm(axis) > 0.0:
        ang = angle(from_vec, to_vec)
        if ang > 0.0:
            return rotquat(normalize(axis), ang)
    elif dot(from_vec, to_vec) < 0.0:
        if orth_axis is None:
            orth_axis = orthvec(normalize(to_vec))
        return rotquat(orth_axis, PI)
    return (1.0, 0.0, 0.0, 0.0)


def alignvecs(
    from_vec1: Vector, from_vec2: Vector, to_vec1: Vector, to_vec2: Vector
) -> Quaternion:
    """
    Return the proper transformation from one pair of orthogonal vectors
    `from_vec1` and `from_vec2` to another pair of orthogonal vectors `to_vec1`
    and `to_vec2`.
    """
    quat1 = alignvec(from_vec1, to_vec1, to_vec2)
    from_vec2 = quatrotate(from_vec2, quat1)
    quat2 = alignvec(from_vec2, to_vec2, to_vec1)
    return quatmulquat(quat2, quat1)


def inertia(vecs: Sequence[Vector]) -> Matrix:
    """
    Calculate the inertia tensor of the points of unit mass with positions
    `vecs`.
    """
    xx = 0.0
    yy = 0.0
    zz = 0.0
    xy = 0.0
    yz = 0.0
    zx = 0.0
    for vec in vecs:
        x = vec[0]
        y = vec[1]
        z = vec[2]
        x_x = x * x
        y_y = y * y
        z_z = z * z
        xx += y_y + z_z
        yy += z_z + x_x
        zz += x_x + y_y
        xy -= x * y
        yz -= y * z
        zx -= z * x
    return (
        (xx, xy, zx),
        (xy, yy, yz),
        (zx, yz, zz),
    )


def symmeig(
    symmmat: Matrix, upper: bool = True, fast: bool = True
) -> Tuple[Vector, Matrix]:
    """
    Calculate the eigenpairs of a symmetric matrix `symmmat` and return a tuple
    of sorted eigenvalues and a tuple of corresponding eigenvectors.  If
    `upper` is enabled, the upper triangular part of the matrix is used.  If
    `fast` is enabled, a faster algorithm is used.
    """
    if EIGH_AVAIL and fast:
        vals_, vecs_ = eigh(symmmat, "U" if upper else "L")
        return (float(vals_[0]), float(vals_[1]), float(vals_[2])), (
            (float(vecs_[0, 0]), float(vecs_[1, 0]), float(vecs_[2, 0])),
            (float(vecs_[0, 1]), float(vecs_[1, 1]), float(vecs_[2, 1])),
            (float(vecs_[0, 2]), float(vecs_[1, 2]), float(vecs_[2, 2])),
        )
    # Givens rotation is faster than Householder transformation and
    # Gram-Schmidt process.
    row0 = symmmat[0]
    row1 = symmmat[1]
    row2 = symmmat[2]
    xx = row0[0]
    yy = row1[1]
    zz = row2[2]
    if upper:
        yx = row0[1]
        zx = row0[2]
        zy = row1[2]
    else:
        yx = row1[0]
        zx = row2[0]
        zy = row2[1]
    # Tridiagonalize the matrix.
    length = copysign(hypot(yx, zx), yx)
    if length != 0.0:
        factor = 1.0 / length
        cos = yx * factor
        sin = zx * factor
        cos_sin = cos * sin
        sin_sin = sin * sin
        cos_cos = cos * cos
        temp = cos_sin * zy
        temp += temp
        zy_ = (cos_cos - sin_sin) * zy + cos_sin * (zz - yy)
        yy_ = cos_cos * yy + temp + sin_sin * zz
        zz_ = sin_sin * yy - temp + cos_cos * zz
        yx = length
        zy = zy_
        yy = yy_
        zz = zz_
    else:
        cos = 1.0
        sin = 0.0
    vec1_x = 1.0
    vec1_y = 0.0
    vec1_z = 0.0
    vec2_x = 0.0
    vec2_y = cos
    vec2_z = sin
    vec3_x = 0.0
    vec3_y = -sin
    vec3_z = cos
    # Perform QR iterations.
    while True:
        xx_ = abs(xx)
        yy_ = abs(yy)
        zz_ = abs(zz)
        if abs(yx) <= EPS * (xx_ + yy_) and abs(zy) <= EPS * (yy_ + zz_):
            break
        # Remove `yx` component with the first transformation.
        length = copysign(hypot(xx, yx), xx)
        transform1 = length != 0.0
        if transform1:
            factor = 1.0 / length
            cos1 = xx * factor
            sin1 = yx * factor
            xx = length
            xy = cos1 * yx + sin1 * yy
            yy = cos1 * yy - sin1 * yx
            yz = cos1 * zy
            temp1 = vec1_x
            temp2 = vec2_x
            vec1_x = cos1 * temp1 + sin1 * temp2
            vec2_x = cos1 * temp2 - sin1 * temp1
            temp1 = vec1_y
            temp2 = vec2_y
            vec1_y = cos1 * temp1 + sin1 * temp2
            vec2_y = cos1 * temp2 - sin1 * temp1
            temp1 = vec1_z
            temp2 = vec2_z
            vec1_z = cos1 * temp1 + sin1 * temp2
            vec2_z = cos1 * temp2 - sin1 * temp1
        else:
            xy = yx
            yz = zy
        # Remove `zy` component with the second transformation.
        length = copysign(hypot(yy, zy), yy)
        transform2 = length != 0.0
        if transform2:
            factor = 1.0 / length
            cos2 = yy * factor
            sin2 = zy * factor
            yy = length
            temp1 = yz
            temp2 = zz
            yz = cos2 * temp1 + sin2 * temp2
            zz = cos2 * temp2 - sin2 * temp1
            temp1 = vec2_x
            temp2 = vec3_x
            vec2_x = cos2 * temp1 + sin2 * temp2
            vec3_x = cos2 * temp2 - sin2 * temp1
            temp1 = vec2_y
            temp2 = vec3_y
            vec2_y = cos2 * temp1 + sin2 * temp2
            vec3_y = cos2 * temp2 - sin2 * temp1
            temp1 = vec2_z
            temp2 = vec3_z
            vec2_z = cos2 * temp1 + sin2 * temp2
            vec3_z = cos2 * temp2 - sin2 * temp1
        # Apply the transpose of the first transformation.
        if transform1:
            temp1 = xx
            temp2 = xy
            xx = cos1 * temp1 + sin1 * temp2
            xy = cos1 * temp2 - sin1 * temp1
            yx = sin1 * yy
            yy = cos1 * yy
        # Apply the transpose of the second transformation.
        if transform2:
            yy = cos2 * yy + sin2 * yz
            zy = sin2 * zz
            zz = cos2 * zz
    vals = (xx, yy, zz)
    vecs = (
        (vec1_x, vec1_y, vec1_z),
        (vec2_x, vec2_y, vec2_z),
        (vec3_x, vec3_y, vec3_z),
    )
    # Sort the eigenvalues and eigenvectors.
    if xx <= yy:
        if yy <= zz:
            order = (0, 1, 2)
        else:
            order = (0, 2, 1) if xx <= zz else (2, 0, 1)
    else:
        if yy <= zz:
            order = (1, 0, 2) if xx <= zz else (1, 2, 0)
        else:
            order = (2, 1, 0)
    i1 = order[0]
    i2 = order[1]
    i3 = order[2]
    return (vals[i1], vals[i2], vals[i3]), (vecs[i1], vecs[i2], vecs[i3])


_Frac = Tuple[int, int]
_Terms = Dict[Tuple[int, int, int, int], int]


def spherfuncs(
    degree: int, expand: bool = False
) -> List[Tuple[_Frac, _Terms, _Terms]]:
    funcs: List[Tuple[_Frac, _Terms, _Terms]] = [((1, 1), {}, {})] * (
        degree + degree + 1
    )
    for order in range(degree + 1):
        zenith_terms = {}
        for i in range((degree - order) // 2 + 1):
            temp1 = i + i
            temp2 = degree - temp1
            coef = (
                comb(degree, i)
                * comb(temp2 + degree, degree)
                * comb(temp2, order)
                * factorial(order)
            )
            if i % 2 != 0:
                coef = -coef
            zenith_terms[(0, 0, temp2 - order, temp1)] = coef
        all_azimuth_terms: Tuple[_Terms, _Terms] = ({}, {})
        for i in range(order + 1):
            coef = comb(order, i)
            if (i // 2) % 2 == 1:
                coef = -coef
            all_azimuth_terms[i % 2][(order - i, i, 0, 0)] = coef
        divisor = gcd(*zenith_terms.values())
        if divisor > 1:
            for pows in zenith_terms:
                zenith_terms[pows] //= divisor
        common_nom = divisor * divisor
        common_denom = 1 << (degree + degree)
        for factor in range(degree - order + 1, degree + order + 1):
            common_denom *= factor
        if order != 0:
            common_nom *= 2
        if expand:
            repeat = True
            while repeat:
                repeat = False
                for pows in tuple(zenith_terms):
                    if pows[3] > 0:
                        repeat = True
                        coef = zenith_terms[pows]
                        del zenith_terms[pows]
                        arr = list(pows)
                        arr[3] -= 2
                        for i in range(3):
                            arr[i] += 2
                            new_pows = (arr[0], arr[1], arr[2], arr[3])
                            arr[i] -= 2
                            if new_pows not in zenith_terms:
                                zenith_terms[new_pows] = 0
                            zenith_terms[new_pows] += coef
        for i, azimuth_terms in enumerate(all_azimuth_terms):
            if len(azimuth_terms) == 0:
                continue
            idx = degree
            if i == 0:
                idx += order
            else:
                idx -= order
            divisor = gcd(*azimuth_terms.values())
            if divisor > 1:
                for pows in azimuth_terms:
                    azimuth_terms[pows] //= divisor
            frac = reducefrac(common_nom * divisor * divisor, common_denom)
            if expand:
                terms: _Terms = {}
                for azimuth_term, azimuth_coef in azimuth_terms.items():
                    azimuth_pows = tuple(azimuth_term)
                    for zenith_term, zenith_coef in zenith_terms.items():
                        zenith_pows = tuple(zenith_term)
                        pows = (
                            azimuth_pows[0] + zenith_pows[0],
                            azimuth_pows[1] + zenith_pows[1],
                            azimuth_pows[2] + zenith_pows[2],
                            azimuth_pows[3] + zenith_pows[3],
                        )
                        if pows not in terms:
                            terms[pows] = 0
                        terms[pows] += azimuth_coef * zenith_coef
                funcs[idx] = (frac, terms, {})
            else:
                funcs[idx] = (frac, azimuth_terms, zenith_terms)
    return funcs


def spherfunclabels(
    degree: int, expand: bool = False, norm: bool = False
) -> List[str]:
    def parenthesize(string: str) -> str:
        return "(" + string + ")" if "+" in string or "-" in string else string

    strings = []
    for func in spherfuncs(degree, expand):
        string = ""
        parenthesized = False
        for terms in func[1:]:
            expr = ""
            for pows, coef in terms.items():
                if coef > 0:
                    expr += "+"
                else:
                    expr += "-"
                    coef = -coef
                if coef > 1:
                    expr += str(coef)
                expr += (
                    "x" * pows[0]
                    + "y" * pows[1]
                    + "z" * pows[2]
                    + "r" * pows[3]
                )
            if expr.startswith("+"):
                expr = expr[1:]
            if len(string) > 0 and len(expr) > 0:
                string = parenthesize(string)
                expr = parenthesize(expr)
                parenthesized = True
            string += expr
        if norm:
            factors = []
            nom, denom = func[0]
            prefactor_nom, radicand_nom = sqrtfactor(nom)
            prefactor_denom, radicand_denom = sqrtfactor(denom)
            for nom, denom, prefix, postfix in (
                (prefactor_nom, prefactor_denom, "", ""),
                (radicand_nom, radicand_denom, "sqrt(", ")"),
            ):
                factor = str(nom)
                if denom > 1:
                    factor += "/" + str(denom)
                if factor != "1":
                    factors.append(prefix + factor + postfix)
            if len(factors) > 0:
                if len(string) > 0:
                    if not parenthesized:
                        string = parenthesize(string)
                    factors.append(string)
                string = "*".join(factors)
        if len(string) == 0:
            string = "1"
        strings.append(string)
    return strings


def realspher(complex_coefs: Sequence[complex]) -> List[float]:
    """
    Convert complex spherical harmonics with coefficients `complex_coefs` to
    real.
    """
    dim = len(complex_coefs)
    degree = dim // 2
    real_coefs = [0.0] * dim
    real_coefs[degree] = complex_coefs[degree].real
    for order in range(1, degree + 1):
        i1 = degree + order
        i2 = degree - order
        coef1 = _SQRT_HALF * complex_coefs[i1]
        coef2 = _SQRT_HALF * complex_coefs[i2]
        if order % 2 == 0:
            real_coefs[i1] = coef2.real + coef1.real
            real_coefs[i2] = coef2.imag - coef1.imag
        else:
            real_coefs[i1] = coef2.real - coef1.real
            real_coefs[i2] = -coef2.imag - coef1.imag
    return real_coefs


def complexspher(real_coefs: Sequence[float]) -> List[complex]:
    """
    Convert real spherical harmonics with coefficients `real_coefs` to complex.
    """
    dim = len(real_coefs)
    degree = dim // 2
    complex_coefs = [0.0j] * dim
    complex_coefs[degree] = complex(real_coefs[degree], 0.0)
    for order in range(1, degree + 1):
        i1 = degree + order
        i2 = degree - order
        coef1 = _SQRT_HALF * real_coefs[i1]
        coef2 = _SQRT_HALF * real_coefs[i2]
        if order % 2 == 0:
            complex_coefs[i1] = complex(coef1, coef2)
        else:
            complex_coefs[i1] = complex(-coef1, -coef2)
        complex_coefs[i2] = complex(coef1, -coef2)
    return complex_coefs


def orthrotmatspher(
    degrees: Sequence[int], angle: float
) -> List[List[List[complex]]]:
    dim = 2 * max(degrees) + 1
    angle *= 0.5
    cosine = cos(angle)
    sine = sin(angle)
    factorial = 1
    factorials = [factorial]
    pow_cos = 1.0
    cosines = [pow_cos]
    pow_sin = 1.0
    sines = [pow_sin]
    for i in range(1, dim):
        factorial *= i
        factorials.append(factorial)
        pow_cos *= cosine
        cosines.append(pow_cos)
        pow_sin *= sine
        sines.append(pow_sin)
    prods = [[1] * dim for _ in range(dim)]
    for i1 in range(dim):
        factorial = factorials[i1]
        for i2 in range(i1, dim):
            prod = factorial * factorials[i2]
            prods[i1][i2] = prod
            prods[i2][i1] = prod
    mats = []
    for degree in degrees:
        dim = degree + degree + 1
        mat = [[0.0j] * dim for _ in range(dim)]
        for order1 in range(-degree, degree + 1):
            pos1 = degree + order1
            neg1 = degree - order1
            factor = prods[pos1][neg1]
            for order2 in range(order1, degree + 1):
                pos2 = degree + order2
                neg2 = degree - order2
                diff = order1 - order2
                total = 0.0
                for i in range(max(0, diff), min(pos1, neg2) + 1):
                    diff1 = pos1 - i
                    diff2 = neg2 - i
                    diff3 = i - diff
                    part = (
                        cosines[diff1 + diff2]
                        * sines[i + diff3]
                        / (prods[diff1][diff2] * prods[i][diff3])
                    )
                    if diff3 % 2 != 0:
                        part = -part
                    total += part
                entry = total * sqrt(prods[pos2][neg2] * factor)
                mat[pos2][pos1] = entry
                if pos1 != pos2:
                    if diff % 2 != 0:
                        entry = -entry
                    mat[pos1][pos2] = entry
        mats.append(mat)
    return mats


def rotmatspher(
    degrees: Sequence[int], quat: Quaternion
) -> List[List[List[complex]]]:
    angles = quatzyzangles(quat)
    mats = orthrotmatspher(degrees, angles[1])
    angle1 = angles[0]
    angle2 = angles[2]
    for degree, mat in zip(degrees, mats):
        dim = degree + degree + 1
        for i1 in range(dim):
            row = mat[i1]
            phase1 = (degree - i1) * angle1
            for i2 in range(dim):
                phase2 = (degree - i2) * angle2
                row[i2] *= exp(complex(0.0, phase1 + phase2))
    return mats
