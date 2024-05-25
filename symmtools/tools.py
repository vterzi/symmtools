"""Miscellaneous functions."""

__all__ = [
    "chcoords",
    "signvar",
    "ax3permut",
    "topoints",
    "generate",
    "str2elems",
]

from re import findall

from .const import ORIGIN, TOL
from .primitive import Point, LabeledPoint, Points
from .transform import Transformation
from .typehints import (
    Optional,
    Sequence,
    List,
    Bool,
    Int,
    Real,
    Ints,
    RealVector,
    RealVectors,
)

_LABEL_RE = r"(?:\b[A-Za-z_]\w*\b)"
_FLOAT_RE = r"(?:[+\-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+\-]?\d+)?)"


def chcoords(
    vecs: RealVectors,
    origin: RealVector = ORIGIN,
    axes: Optional[Ints] = None,
) -> List[List[Real]]:
    """
    Change the coordinate system of vectors `vecs` to a coordinate system with
    an origin `origin` and an axes order `axes`.  If `axes` is `None`, the
    original axes order is used.
    """
    if axes is None:
        axes = tuple(range(len(origin)))
    origin = tuple(-coord for coord in origin)
    res = []
    for vec in vecs:
        new = [*origin]
        for ax, coord in zip(axes, vec):
            new[ax] += coord
        res.append(new)
    return res


def signvar(
    vec: RealVector, parity: Int = 0, unique: Bool = False
) -> List[List[Real]]:
    """
    Generate vectors with all possible sign changes of the coordinates of a
    vector `vec` that satisfy a parity `parity`.  If `parity` is
    positive/negative, only the vectors resulting from even/odd number of sign
    changes are returned.  If `parity` is zero, all vectors are returned.  If
    `unique` is `True`, only linearly independent vectors are returned.
    """
    res = []
    for n in range(2 ** len(vec)):
        new = [*vec]
        sign = 1
        i = 0
        while n > 0:
            if n % 2 == 1:
                new[i] *= -1
                sign *= -1
            n //= 2
            i += 1
        if (sign * parity >= 0 and new not in res) and not (
            unique and [-coord for coord in new] in res
        ):
            res.append(new)
    return res


def ax3permut(vecs: RealVectors) -> List[List[Real]]:
    """
    Generate all possible vectors by applying a circular permutation on the
    coordinates of 3D vectors `vecs`.
    """
    vecs = chcoords(vecs)
    res = []
    for i in range(3):
        for vec in vecs:
            res.append([vec[i % 3], vec[(i + 1) % 3], vec[(i + 2) % 3]])
    return res


def topoints(points: RealVectors) -> List[Point]:
    """
    Convert a sequence of points `points` to a sequence of `Point` instances.
    """
    return [Point(point) for point in points]


def generate(
    points: Sequence[Point],
    transformations: Sequence[Transformation] = (),
    tol: float = TOL,
) -> Points:
    """
    Generate all unique points by applying transformations `transformations` to
    points `points` and return them as a `Points` instance.
    """
    points = list(points)
    fi = 0
    li = len(points)
    while fi < li:
        for transformation in transformations:
            for i in range(fi, li):
                point = transformation(points[i])
                new = True
                for ref_point in points:
                    if point.same(ref_point, tol):
                        new = False
                        break
                if new:
                    points.append(point)
        fi = li
        li = len(points)
    return Points(points)


def str2elems(string: str) -> Points:
    """
    Convert a string `string` to a `Points` instance.  Each three consecutive
    floating-point numbers are parsed as a `Point` instance.  If they are
    preceded by a label satisfying the rules of variable names, a
    `LabeledPoint` instance is created instead.
    """
    points = []
    for match in findall(
        rf"(?:({_LABEL_RE})\s+)?({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})",
        string,
    ):
        label, x, y, z = match
        points.append(
            LabeledPoint((x, y, z), label) if label else Point((x, y, z))
        )
    return Points(points)
