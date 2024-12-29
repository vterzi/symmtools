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
    "matmul",
    "rmatmul",
    "matprod",
    "trigrotate",
    "rotate",
    "reflect",
    "trigrotmat",
    "rotmat",
    "reflmat",
    "inertia",
    "symmeig",
]

from math import copysign, hypot, sqrt, sin, cos, acos
from typing import Sequence, Tuple

try:
    from numpy.linalg import eigh
except ImportError:
    pass

from .const import PI, PI_2, EPS
from .utils import clamp

# `numpy` with `numpy.ndarray` is slower than `math` with `tuple`
# array unpacking is slower than indexing

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
    return mul(vec, 1.0 / norm(vec))


def orthogonalize(vec: Vector, unitvec: Vector) -> Vector:
    """Orthogonalize a vector `vec` to a unit vector `unitvec`."""
    # `sub(vec, mul(unitvec, dot(vec, unitvec)))` is slower
    x1 = vec[0]
    y1 = vec[1]
    z1 = vec[2]
    x2 = unitvec[0]
    y2 = unitvec[1]
    z2 = unitvec[2]
    prod = x1 * x2 + y1 * y2 + z1 * z2
    return (x1 - prod * x2, y1 - prod * y2, z1 - prod * z2)


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
    # `norm(sub(vec1, vec2))` is slower
    # `max(sub(vec1, vec2))` is slower
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
    # `norm(cross(vec1, vec2))` is slower
    # `max(cross(vec1, vec2))` is slower
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
    # `abs(abs(dot(unitvec1, unitvec2)) - 1)` is faster but less accurate
    # `min(diff(unitvec1, unitvec2), diff(unitvec1, neg(unitvec2)))` is slower
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
    odd indices, and orthogonalizing the result to `unitvec`.
    """
    # `normalize(orthogonalize((-unitvec[2], unitvec[0], -unitvec[1]),
    # unitvec))` is slower
    x2 = unitvec[0]
    y2 = unitvec[1]
    z2 = unitvec[2]
    x1 = -z2
    y1 = x2
    z1 = -y2
    prod = x1 * x2 + y1 * y2 + z1 * z2
    x1 -= prod * x2
    y1 -= prod * y2
    z1 -= prod * z2
    scalar = 1.0 / hypot(x1, y1, z1)
    return (x1 * scalar, y1 * scalar, z1 * scalar)


def angle(vec1: Vector, vec2: Vector) -> float:
    """Calculate the angle between two vectors `vec1` and `vec2`."""
    # `acos(clamp(dot(unitvec1, unitvec2), -1.0, 1.0))` is less accurate
    # `acos(clamp(dot(vec1, vec2) / sqrt(sqnorm(vec1) * sqnorm(vec2)), -1.0,
    # 1.0))` is slower
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
    r0 = mat[0]
    r1 = mat[1]
    r2 = mat[2]
    return (
        (r0[0], r1[0], r2[0]),
        (r0[1], r1[1], r2[1]),
        (r0[2], r1[2], r2[2]),
    )


def matmul(mat: Matrix, vec: Vector) -> Vector:
    """Multiply a matrix `mat` by a vector `vec`."""
    r0 = mat[0]
    r1 = mat[1]
    r2 = mat[2]
    x = vec[0]
    y = vec[1]
    z = vec[2]
    return (
        r0[0] * x + r0[1] * y + r0[2] * z,
        r1[0] * x + r1[1] * y + r1[2] * z,
        r2[0] * x + r2[1] * y + r2[2] * z,
    )


def rmatmul(vec: Vector, mat: Matrix) -> Vector:
    """Multiply a vector `vec` by a matrix `mat`."""
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r0 = mat[0]
    r1 = mat[1]
    r2 = mat[2]
    return (
        x * r0[0] + y * r1[0] + z * r2[0],
        x * r0[1] + y * r1[1] + z * r2[1],
        x * r0[2] + y * r1[2] + z * r2[2],
    )


def matprod(mat1: Matrix, mat2: Matrix) -> Matrix:
    """Multiply two matrices `mat1` and `mat2`."""
    lt0 = mat1[0]
    lt1 = mat1[1]
    lt2 = mat1[2]
    rt0 = mat2[0]
    rt1 = mat2[1]
    rt2 = mat2[2]
    xx1 = lt0[0]
    xy1 = lt0[1]
    xz1 = lt0[2]
    yx1 = lt1[0]
    yy1 = lt1[1]
    yz1 = lt1[2]
    zx1 = lt2[0]
    zy1 = lt2[1]
    zz1 = lt2[2]
    xx2 = rt0[0]
    xy2 = rt0[1]
    xz2 = rt0[2]
    yx2 = rt1[0]
    yy2 = rt1[1]
    yz2 = rt1[2]
    zx2 = rt2[0]
    zy2 = rt2[1]
    zz2 = rt2[2]
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


def trigrotate(vec: Vector, normal: Vector, cos: float, sin: float) -> Vector:
    """
    Rotate a vector `vec` by an angle with cosine `cos` and sine `sin` around
    an axis that contains the origin and is described by a unit vector `axis`.
    """
    # ```
    # base = mul(normal, dot(vec, normal))
    # projection = sub(vec, base)
    # perpendicular = cross(normal, projection)
    # add(add(base, mul(projection, cos)), mul(perpendicular, sin))
    # ``` is slower
    vec_x = vec[0]
    vec_y = vec[1]
    vec_z = vec[2]
    normal_x = normal[0]
    normal_y = normal[1]
    normal_z = normal[2]
    prod = vec_x * normal_x + vec_y * normal_y + vec_z * normal_z
    base_x = normal_x * prod
    base_y = normal_y * prod
    base_z = normal_z * prod
    projection_x = vec_x - base_x
    projection_y = vec_y - base_y
    projection_z = vec_z - base_z
    perpendicular_x = normal_y * projection_z - normal_z * projection_y
    perpendicular_y = normal_z * projection_x - normal_x * projection_z
    perpendicular_z = normal_x * projection_y - normal_y * projection_x
    return (
        base_x + projection_x * cos + perpendicular_x * sin,
        base_y + projection_y * cos + perpendicular_y * sin,
        base_z + projection_z * cos + perpendicular_z * sin,
    )


def rotate(vec: Vector, normal: Vector, angle: float) -> Vector:
    """
    Rotate a vector `vec` by an angle `angle` around an axis that contains the
    origin and is described by a unit vector `axis`.
    """
    return trigrotate(vec, normal, cos(angle), sin(angle))


def reflect(vec: Vector, normal: Vector) -> Vector:
    """
    Reflect a vector `vec` through a plane that contains the origin and whose
    normal is described by a unit vector `normal`.
    """
    # `sub(vec, mul(normal, 2.0 * dot(vec, normal)))` is slower
    vec_x = vec[0]
    vec_y = vec[1]
    vec_z = vec[2]
    normal_x = normal[0]
    normal_y = normal[1]
    normal_z = normal[2]
    prod = 2.0 * (vec_x * normal_x + vec_y * normal_y + vec_z * normal_z)
    return (
        vec_x - normal_x * prod,
        vec_y - normal_y * prod,
        vec_z - normal_z * prod,
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
    xc = x * (1.0 - cos)
    yc = y * (1.0 - cos)
    zc = z * (1.0 - cos)
    xs = x * sin
    ys = y * sin
    zs = z * sin
    xyc = x * yc
    yzc = y * zc
    zxc = z * xc
    return (
        (cos + x * xc, xyc - zs, zxc + ys),
        (xyc + zs, cos + y * yc, yzc - xs),
        (zxc - ys, yzc + xs, cos + z * zc),
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
    dx = x + x
    dy = y + y
    dz = z + z
    xy = -x * dy
    yz = -y * dz
    zx = -z * dx
    return (
        (1.0 - x * dx, xy, zx),
        (xy, 1.0 - y * dy, yz),
        (zx, yz, 1.0 - z * dz),
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
        xs = x * x
        ys = y * y
        zs = z * z
        xx += ys + zs
        yy += zs + xs
        zz += xs + ys
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
    if eigh in globals() and fast:
        vals_, vecs_ = eigh(symmmat, "U" if upper else "L")
        vals = (float(vals_[0]), float(vals_[1]), float(vals_[2]))
        vecs = (
            (float(vecs_[0, 0]), float(vecs_[1, 0]), float(vecs_[2, 0])),
            (float(vecs_[0, 1]), float(vecs_[1, 1]), float(vecs_[2, 1])),
            (float(vecs_[0, 2]), float(vecs_[1, 2]), float(vecs_[2, 2])),
        )
        return vals, vecs
    r0 = symmmat[0]
    r1 = symmmat[1]
    r2 = symmmat[2]
    xx = r0[0]
    yy = r1[1]
    zz = r2[2]
    if upper:
        yx = r0[1]
        zx = r0[2]
        zy = r1[2]
    else:
        yx = r1[0]
        zx = r2[0]
        zy = r2[1]
    length = copysign(hypot(yx, zx), yx)
    if length != 0.0:
        factor = 1.0 / length
        cos = yx * factor
        sin = zx * factor
        cos_sin = cos * sin
        sin_sin = sin * sin
        cos_cos = cos * cos
        summand = cos_sin * zy
        summand += summand
        zy_ = (cos_cos - sin_sin) * zy + cos_sin * (zz - yy)
        yy_ = cos_cos * yy + summand + sin_sin * zz
        zz_ = sin_sin * yy - summand + cos_cos * zz
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
    while True:
        yy_ = abs(yy)
        if abs(yx) <= EPS * (abs(xx) + yy_) and abs(zy) <= EPS * (
            yy_ + abs(zz)
        ):
            break
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
        if transform1:
            temp1 = xx
            temp2 = xy
            xx = cos1 * temp1 + sin1 * temp2
            xy = cos1 * temp2 - sin1 * temp1
            yx = sin1 * yy
            yy = cos1 * yy
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
