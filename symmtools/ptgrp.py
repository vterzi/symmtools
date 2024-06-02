"""Class for point groups."""

__all__ = ["symmelems", "ptgrp", "symb2symmelems", "PointGroup"]

from numpy import cross

from .const import INF, PHI, TOL, PRIMAX, SECAX, INF_SYMB
from .tools import signvar, ax3permut
from .transform import Transformable, Transformation, Identity
from .symmelem import (
    SymmetryElement,
    InversionCenter,
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    CenterRotationAxes,
    AxisReflectionPlanes,
    CenterReflectionPlanes,
    CenterRotoreflectionAxes,
)
from .primitive import Points
from .vecop import norm, parallel, unitparallel, perpendicular
from .typehints import Any, Union, Sequence, Tuple, List, Vector

_RotationAxis = Union[RotationAxis, InfRotationAxis]
_ReflectionPlane = ReflectionPlane
_RotoreflectionAxis = Union[RotoreflectionAxis, InfRotoreflectionAxis]


def symmelems(points: Points, tol: float = TOL) -> Tuple[
    int,
    bool,
    Sequence[_RotationAxis],
    Sequence[_ReflectionPlane],
    Sequence[_RotoreflectionAxis],
]:
    """
    Determine dimensionality, ivertibility, rotation axes, reflection planes,
    and rotoreflection axes of a set of points `points` within a tolerance
    `tol`.
    """

    def contains(array: List[Vector], vector: Vector) -> bool:
        for elem in array:
            if unitparallel(elem, vector, tol):
                return True
        array.append(vector)
        return False

    def add_rotation(vector: Vector, order: int) -> bool:
        rotation = RotationAxis(vector, order)
        if rotation.symmetric(points, tol):
            i = 0
            while i < len(rotations) and rotations[i].order > order:
                i += 1
            rotations.insert(i, rotation)
            return True
        return False

    def add_reflection(vector: Vector) -> bool:
        reflection = ReflectionPlane(vector)
        if reflection.symmetric(points, tol):
            reflections.append(reflection)
            return True
        return False

    def add_rotoreflection(vector: Vector, order: int) -> bool:
        for factor in (2, 1):
            order_ = factor * order
            if order_ > 2:
                rotoreflection = RotoreflectionAxis(vector, order_)
                if rotoreflection.symmetric(points, tol):
                    i = 0
                    while (
                        i < len(rotoreflections)
                        and rotoreflections[i].order > order_
                    ):
                        i += 1
                    rotoreflections.insert(i, rotoreflection)
                    return True
        return False

    if not points.nondegen(tol):
        raise ValueError(
            "at least two identical elements in the instance of "
            + points.__class__.__name__
            + " for the given tolerance"
        )
    dim = 3
    invertible = InversionCenter().symmetric(points, tol)
    rotations: List[_RotationAxis] = []
    reflections: List[_ReflectionPlane] = []
    rotoreflections: List[_RotoreflectionAxis] = []
    n_points = len(points)
    if n_points == 1:
        dim = 0
    axes: List[Vector] = []
    planes: List[Vector] = []
    direction = None
    collinear = True
    for i1 in range(n_points - 1):
        # for all point pairs
        for i2 in range(i1 + 1, n_points):
            # for all point triplets
            for i3 in range(i2 + 1, n_points):
                # find the normal of the plane containing the point triplet
                normal: Vector = cross(
                    points[i2].pos - points[i1].pos,
                    points[i3].pos - points[i1].pos,
                )
                normal_norm = norm(normal)
                # if the point triplet is collinear
                if normal_norm <= tol:
                    continue
                # not all points are collinear
                collinear = False
                rotation = normal / normal_norm
                if not contains(axes, rotation):
                    # calculate the distance between the origin and the plane
                    dist = points[i1].pos.dot(rotation)
                    # initial number of points in the plane
                    max_order = 3
                    # for other points
                    for i4 in range(i3 + 1, n_points):
                        # if the point is in the plane
                        if abs(points[i4].pos.dot(rotation) - dist) <= tol:
                            # increase the number of points in the plane
                            max_order += 1
                    if (
                        max_order == n_points
                        and direction is None
                        and abs(dist) <= tol
                    ):
                        # all points are coplanar
                        dim = 2
                        direction = rotation
                        # add reflection (described by the plane containing
                        #   all points)
                        reflections.append(ReflectionPlane(rotation))
                    # for all possible orders > 2 starting from the highest
                    for order in range(max_order, 2, -1):
                        if add_rotoreflection(rotation, order):
                            break
                    # for all possible orders > 1 starting from the highest
                    for order in range(max_order, 1, -1):
                        if add_rotation(rotation, order):
                            break
            # directed segment between the point pair
            segment = points[i1].pos - points[i2].pos
            # midpoint of the segment
            midpoint = 0.5 * (points[i1].pos + points[i2].pos)
            reflection = segment / norm(segment)
            # add rotation with order infinity
            if (
                collinear
                and direction is None
                and parallel(points[i1].pos, points[i2].pos, tol)
            ):
                dim = 1
                direction = reflection
                rotations.insert(0, InfRotationAxis(reflection))
                if invertible:
                    rotoreflections.insert(
                        0, InfRotoreflectionAxis(reflection)
                    )
            # if the distance from the origin to the segment doesn't divide it
            #   in halves
            if not perpendicular(segment, midpoint, tol):
                continue
            midpoint_norm = norm(midpoint)
            # if the distance from the origin to the midpoint is not zero or
            #   all points are in one plane
            if midpoint_norm > tol or dim == 2:
                rotation = (
                    midpoint / midpoint_norm
                    if direction is None
                    else cross(direction, reflection)
                )
                if not contains(axes, rotation):
                    order = 2
                    add_rotoreflection(rotation, order)
                    add_rotation(rotation, order)
            if not contains(planes, reflection):
                add_reflection(reflection)
    return (
        dim,
        invertible,
        tuple(rotations),
        tuple(reflections),
        tuple(rotoreflections),
    )


def ptgrp(points: Points, tol: float = TOL) -> str:
    """
    Determine the point group symbol of a set of points `points` within a
    tolerance `tol`.
    """
    dim, invertible, rotations, reflections, rotoreflections = symmelems(
        points, tol
    )
    if dim == 0:
        return "Kh"  # 'K'
    if dim == 1:
        return f"D{INF_SYMB}h" if invertible else f"C{INF_SYMB}v"
    sigma = len(reflections) > 0
    if len(rotations) == 0:
        if sigma:
            return "Cs"
        if invertible:
            return "Ci"
        return "C1"
    rotation = rotations[0]
    order = rotations[0].order
    h = False
    if sigma:
        for reflection in reflections:
            if unitparallel(rotation.vec, reflection.vec, tol):
                h = True
                break
    if len(rotations) >= 2:
        if order > 2 and rotations[1].order > 2:
            if order == 5:
                return "Ih" if sigma else "I"
            if order == 4:
                return "Oh" if sigma else "O"
            return ("Th" if h else "Td") if sigma else "T"
        return (f"D{order}h" if h else f"D{order}d") if sigma else f"D{order}"
    else:
        if sigma:
            return f"C{order}h" if h else f"C{order}v"
        if len(rotoreflections) > 0:
            return f"S{2*order}"
        return f"C{order}"


def symb2symmelems(
    symb: str,
) -> Tuple[str, Sequence[SymmetryElement], Sequence[str]]:
    """
    Return the standardized symbol, all symmetry elements in the standardized
    space orientation, and their labels for a point group with a symbol `symb`.
    """
    if not symb:
        raise ValueError("empty symbol")
    rotation = symb[0]
    subscript = symb[1:]
    if subscript.startswith(INF_SYMB):
        i = len(INF_SYMB)
        order = INF_SYMB
        n = 0
        inf = True
        reflection = subscript[i:]
    else:
        length = len(subscript)
        i = 0
        while i < length and subscript[i].isdigit():
            i += 1
        order = subscript[:i]
        if i > 0:
            if order.startswith("0"):
                raise ValueError("leading zero in the order of the symbol")
            n = int(order)
        else:
            n = 0
        inf = False
        reflection = subscript[i:]
    symmelems: List[SymmetryElement] = []
    labels: List[str] = []

    def add(symmelem: SymmetryElement, label: str = "") -> None:
        symmelems.append(symmelem)
        labels.append(label)

    if rotation == "C":
        if order:
            if not reflection:
                if n > 1:
                    add(RotationAxis(PRIMAX, n))
            elif reflection == "i":
                if n == 1:
                    return symb2symmelems("Ci")
                elif n % 2 == 1:
                    return symb2symmelems(f"S{2 * n}")
                elif (n // 2) % 2 == 1:
                    return symb2symmelems(f"S{n // 2}")
                else:
                    return symb2symmelems(f"S{order}")
            elif reflection == "v":
                if n == 1:
                    return symb2symmelems("Cs")
                else:
                    if not inf:
                        add(RotationAxis(PRIMAX, n))
                        plane = ReflectionPlane(SECAX)
                        transforms = RotationAxis(
                            PRIMAX, 2 * n
                        ).transformations()
                        planes = (plane,) + tuple(
                            transform(plane)
                            for transform in transforms[: n - 1]
                        )
                        if n % 2 == 1:
                            for plane in planes:
                                add(plane, "v")
                        else:
                            n_planes = len(planes)
                            for i in range(0, n_planes, 2):
                                add(planes[i], "v")
                            for i in range(1, n_planes, 2):
                                add(planes[i], "d")
                            add(InversionCenter())
                    else:
                        add(InfRotationAxis(PRIMAX))
                        add(AxisReflectionPlanes(SECAX), "v")
            elif reflection == "h":
                if n == 1:
                    return symb2symmelems("Cs")
                else:
                    if not inf:
                        add(RotationAxis(PRIMAX, n))
                        add(ReflectionPlane(PRIMAX), "h")
                        if n % 2 == 0:
                            add(InversionCenter())
                        if n > 2:
                            add(RotoreflectionAxis(PRIMAX, n))
                    else:
                        add(InfRotationAxis(PRIMAX))
                        add(ReflectionPlane(PRIMAX), "h")
                        add(InversionCenter())
                        add(InfRotoreflectionAxis(PRIMAX))
            else:
                raise ValueError(
                    "a symbol starting with 'C' and an order can end only with"
                    + " '', 'i', 'v', or 'h'"
                )
        elif reflection == "s":
            add(ReflectionPlane(PRIMAX))
        elif reflection == "i":
            add(InversionCenter())
        else:
            raise ValueError(
                "a symbol starting with 'C' should have an order or end with"
                + " 's' or 'i'"
            )
    elif rotation == "S":
        if reflection:
            raise ValueError(
                "a symbol starting with 'S' can end only with an order"
            )
        if n % 2 == 1 or inf:
            return symb2symmelems(f"C{order}h")
        elif n == 2:
            return symb2symmelems("Ci")
        elif n > 0:
            add(RotoreflectionAxis(PRIMAX, n))
            add(RotationAxis(PRIMAX, n // 2))
            if (n // 2) % 2 == 1:
                add(InversionCenter())
        else:
            raise ValueError("a symbol starting with 'S' should have an order")
    elif rotation == "D":
        if n > 0:
            add(RotationAxis(PRIMAX, n))
            transforms = RotationAxis(PRIMAX, 2 * n).transformations()
            axis = RotationAxis(SECAX, 2)
            axes = (axis,) + tuple(
                transform(axis) for transform in transforms[: n - 1]
            )
            if n % 2 == 1:
                for axis in axes:
                    add(axis, "'")
            else:
                n_axes = len(axes)
                for i in range(0, n_axes, 2):
                    add(axes[i], "'")
                for i in range(0, n_axes, 2):
                    add(axes[i], "''")
        elif inf:
            add(InfRotationAxis(PRIMAX))
            add(AxisRotationAxes(PRIMAX))
        else:
            raise ValueError("a symbol starting with 'D' should have an order")
        if not reflection:
            if n == 1:
                return symb2symmelems(f"C{2 * n}")
            elif inf:
                add(InversionCenter())
        elif reflection == "d":
            if n == 1:
                return symb2symmelems(f"C{2 * n}h")
            elif inf:
                return symb2symmelems(f"D{order}h")
            else:
                plane = ReflectionPlane(SECAX)
                transforms = RotationAxis(PRIMAX, 2 * n).transformations()
                planes = tuple(transforms[i](plane) for i in range(1, n, 2))
                for plane in planes:
                    add(plane, "d")
                if n % 2 == 1:
                    add(InversionCenter())
                add(RotoreflectionAxis(PRIMAX, 2 * n))
        elif reflection == "h":
            if n == 1:
                return symb2symmelems(f"C{2 * n}v")
            else:
                add(ReflectionPlane(PRIMAX), "h")
                if not inf:
                    plane = ReflectionPlane(SECAX)
                    transforms = RotationAxis(PRIMAX, 2 * n).transformations()
                    planes = (plane,) + tuple(
                        transform(plane) for transform in transforms[: n - 1]
                    )
                    if n % 2 == 1:
                        for plane in planes:
                            add(plane, "v")
                    else:
                        n_planes = len(planes)
                        for i in range(0, n_planes, 2):
                            add(planes[i], "v")
                        for i in range(1, n_planes, 2):
                            add(planes[i], "d")
                        add(InversionCenter())
                    if n > 2:
                        add(RotoreflectionAxis(PRIMAX, n))
                else:
                    add(AxisReflectionPlanes(PRIMAX), "v")
    elif order:
        raise ValueError(
            "only the symbols starting with 'C', 'S', or 'D' can have an order"
        )
    elif rotation == "T":
        vecs3 = signvar([1, 1, 1], 1)
        vecs2 = ax3permut([[1]])
        for n, vecs in ((3, vecs3), (2, vecs2)):
            for vec in vecs:
                add(RotationAxis(vec, n))
        if reflection == "d":
            for vec in ax3permut(signvar([1, 1], 0, True)):
                add(ReflectionPlane(vec), "d")
            n = 4
            for vec in vecs2:
                add(RotoreflectionAxis(vec, n))
        elif reflection == "h":
            for vec in vecs2:
                add(ReflectionPlane(vec), "h")
            add(InversionCenter())
            n = 6
            for vec in vecs3:
                add(RotoreflectionAxis(vec, n))
        else:
            raise ValueError(
                "a symbol starting with 'T' can end only with '', 'd', or 'h'"
            )
    elif rotation == "O":
        vecs4 = ax3permut([[1]])
        vecs3 = signvar([1, 1, 1], 1)
        vecs2 = ax3permut(signvar([1, 1], 0, True))
        for n, vecs in ((4, vecs4), (3, vecs3), (2, vecs2)):
            for vec in vecs:
                add(RotationAxis(vec, n))
        if reflection == "h":
            for vec in vecs4:
                add(ReflectionPlane(vec), "h")
            for vec in vecs2:
                add(ReflectionPlane(vec), "d")
            add(InversionCenter())
            for n, vecs in ((6, vecs3), (4, vecs4)):
                for vec in vecs:
                    add(RotoreflectionAxis(vec, n))
        else:
            raise ValueError(
                "a symbol starting with 'O' can end only with '' or 'h'"
            )
    elif rotation == "I":
        vecs5 = ax3permut(signvar([PHI, 1], 0, True))
        vecs3 = signvar([1, 1, 1], 1) + ax3permut(
            signvar([1, 1 + PHI], 0, True)
        )
        vecs2 = ax3permut([[1], *signvar([1, PHI, 1 + PHI], 0, True)])
        for n, vecs in ((5, vecs5), (3, vecs3), (2, vecs2)):
            for vec in vecs:
                add(RotationAxis(vec, n))
        if reflection == "h":
            for vec in vecs2:
                add(ReflectionPlane(vec))
            add(InversionCenter())
            for n, vecs in ((10, vecs5), (6, vecs3)):
                for vec in vecs:
                    add(RotoreflectionAxis(vec, n))
        else:
            raise ValueError(
                "a symbol starting with 'I' can end only with '' or 'h'"
            )
    elif rotation == "K":
        add(CenterRotationAxes())
        if reflection == "h":
            add(CenterReflectionPlanes())
            add(InversionCenter())
            add(CenterRotoreflectionAxes())
        else:
            raise ValueError(
                "a symbol starting with 'K' can end only with '' or 'h'"
            )
    else:
        raise ValueError(
            "a symbol can start only with 'C', 'S', 'D', 'T', 'O', 'I', or"
            + " 'K'"
        )
    return symb, tuple(symmelems), tuple(labels)


class PointGroup(Transformable):
    """Point group."""

    def __init__(
        self, symb: str, transformation: Transformation = Identity()
    ) -> None:
        """
        Initialize the instance with a symbol `symb` and a transformation
        `transformation` describing the orientation in space.
        """
        symb, symmelems, labels = symb2symmelems(symb)
        self._symb = symb
        self._symmelems = tuple(
            transformation(symmelem) for symmelem in symmelems
        )
        self._labels = labels
        self._transformation = transformation

    @property
    def symb(self) -> str:
        """Return the symbol."""
        return self._symb

    @property
    def symmelems(self) -> Sequence[SymmetryElement]:
        """Return the symmetry elements."""
        return self._symmelems

    @property
    def transformation(self) -> Transformation:
        """Return the transformation describing the orientation in space."""
        return self._transformation

    def args(self) -> str:
        res = self._symb
        if not isinstance(self._transformation, Identity):
            res += f",{self._transformation}"
        return res

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            if self._symb != obj.symb:
                res = INF
            else:
                res = max(res, self._transformation.diff(obj.transformation))
        return res
