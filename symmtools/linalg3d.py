"""Linear algebra functions for 3D vectors and matrices."""

__all__ = [
    "Vector",
    "Matrix",
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
    "inertia",
    "symmeig",
]

from math import copysign, sqrt, hypot, cos, sin, acos
from typing import Sequence, Tuple

try:
    from numpy.linalg import eigh

    EIGH_AVAIL = True
except ImportError:
    EIGH_AVAIL = False

from .const import PI, PI_2, EPS
from .utils import clamp

# `numpy` with `numpy.ndarray` is slower than `math` with `tuple`.
# Array unpacking is slower than indexing.

Vector = Tuple[float, float, float]
Matrix = Tuple[Vector, Vector, Vector]


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
    return div(vec, norm(vec))


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
    Generate a unit vector that is orthogonal to a unit vector `unitvec` by
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
    if ang > PI_2:
        ang = PI - ang
    return ang


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
    Generate a transformation matrix for a rotation by an angle with cosine
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
    Generate a transformation matrix for a rotation by an angle `angle` around
    an axis that contains the origin and is described by a unit vector `axis`.
    """
    return trigrotmat(axis, cos(angle), sin(angle))


def reflmat(normal: Vector) -> Matrix:
    """
    Generate a transformation matrix for a reflection through a plane that
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


def inertia(vecs: Sequence[Vector]) -> Matrix:
    """
    Calculate the inertia tensor of the points of unit mass with positions
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
        x_x = x * x
        y_y = y * y
        z_z = z * z
        xx += y_y + z_z
        yy += z_z + x_x
        zz += x_x + y_y
        xy -= x * y
        zx -= x * z
        yz -= y * z
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
