"""Class for point groups."""

__all__ = ["symmelems", "ptgrp", "PointGroupInfo", "PointGroup"]

from math import sin, cos

from .const import (
    INF,
    PI,
    PHI,
    TOL,
    SPECIAL_ANGLES,
    SYMB,
    ROT_SYMBS,
    REFL_SYMBS,
)
from .vecop import (
    vector,
    norm,
    cross,
    parallel,
    unitparallel,
    perpendicular,
    intersectangle,
)
from .tools import rational, signvar, ax3permut
from .transform import (
    Transformable,
    Transformation,
    Identity,
    Translation,
    Rotation,
    Reflection,
    Rotoreflection,
)
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
    symmelems2nums,
)
from .primitive import Points
from .typehints import (
    TypeVar,
    Type,
    Any,
    Union,
    Sequence,
    Set,
    Tuple,
    List,
    Dict,
    Vector,
    RealVector,
)

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
    if not points.nondegen(tol):
        raise ValueError(
            "at least two identical elements in the instance of "
            + points.__class__.__name__
            + " for the given tolerance"
        )
    rotations: List[_RotationAxis] = []
    reflections: List[_ReflectionPlane] = []
    rotoreflections: List[_RotoreflectionAxis] = []

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

    invertible = InversionCenter().symmetric(points, tol)
    axes: List[Vector] = []
    planes: List[Vector] = []
    direction = None
    first = True
    collinear = False
    coplanar = False
    n_points = len(points)
    for i1 in range(n_points - 1):
        point1 = points[i1]
        pos1 = point1.pos
        for i2 in range(i1 + 1, n_points):
            point2 = points[i2]
            if not point2.similar(point1):
                continue
            pos2 = point2.pos
            segment = pos1 - pos2
            if not collinear:
                if first:
                    first = False
                    collinear = True
                for i3 in range(i2 + 1, n_points):
                    point3 = points[i3]
                    if not point3.similar(point2):
                        continue
                    pos3 = point3.pos
                    normal = cross(segment, pos1 - pos3)
                    normal_norm = norm(normal)
                    if normal_norm <= tol:
                        continue
                    collinear = False
                    rotation = normal / normal_norm
                    if not contains(axes, rotation):
                        dist = pos1.dot(rotation)
                        max_order = 3
                        for i4 in range(i3 + 1, n_points):
                            if abs(points[i4].pos.dot(rotation) - dist) <= tol:
                                max_order += 1
                        if (
                            max_order == n_points
                            and direction is None
                            and abs(dist) <= tol
                        ):
                            coplanar = True
                            direction = rotation
                            reflections.append(ReflectionPlane(rotation))
                        for order in range(max_order, 2, -1):
                            if add_rotoreflection(rotation, order):
                                break
                        for order in range(max_order, 1, -1):
                            if add_rotation(rotation, order):
                                break
            midpoint = 0.5 * (pos1 + pos2)
            reflection = segment / norm(segment)
            if collinear and direction is None and parallel(pos1, pos2, tol):
                direction = reflection
                rotations.insert(0, InfRotationAxis(reflection))
                if invertible:
                    rotoreflections.insert(
                        0, InfRotoreflectionAxis(reflection)
                    )
            if not perpendicular(segment, midpoint, tol):
                continue
            midpoint_norm = norm(midpoint)
            if midpoint_norm > tol or coplanar:
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
    if n_points == 1:
        dim = 0
    elif collinear:
        dim = 1
    elif coplanar:
        dim = 2
    else:
        dim = 3
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
        return f"D{SYMB.inf}h" if invertible else f"C{SYMB.inf}v"
    sigma = len(reflections) > 0
    if len(rotations) == 0:
        if sigma:
            return "Cs"
        if invertible:
            return "Ci"
        return "C1"
    rotation = rotations[0]
    order = rotation.order
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


_DirectionSymmetryElement = Union[
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    AxisReflectionPlanes,
]
_DirectionSymmetryElements = (
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    AxisReflectionPlanes,
)


class PointGroupInfo:
    """
    Point group information containing types and numbers of symmetry elements
    and angles between their axes or normals.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._included: List[_DirectionSymmetryElement] = []
        self._excluded: List[_DirectionSymmetryElement] = []
        self._nums: Dict[int, int] = {}
        self._angles: Dict[Tuple[int, int], Dict[float, int]] = {}

    @property
    def included(self) -> Sequence[_DirectionSymmetryElement]:
        """Return the included symmetry elements."""
        return self._included

    @property
    def excluded(self) -> Sequence[_DirectionSymmetryElement]:
        """Return the excluded symmetry elements."""
        return self._excluded

    @property
    def nums(self) -> Dict[int, int]:
        """Return the types and numbers of symmetry elements."""
        return self._nums

    @property
    def angles(self) -> Dict[Tuple[int, int], Dict[float, int]]:
        """Return the angles between axes or normals of symmetry elements."""
        return self._angles

    def include(
        self,
        symmelems: Union[SymmetryElement, Sequence[SymmetryElement]],
        tol: float,
    ) -> None:
        """
        Include information of one or multiple symmetry elements `symmelems`
        using a tolerance `tol` to calculate exact intersection angles.
        """
        if not isinstance(symmelems, Sequence):
            symmelems = (symmelems,)
        for symmelem1 in symmelems:
            id1 = symmelem1.id
            if isinstance(symmelem1, _DirectionSymmetryElements):
                vec1 = symmelem1.vec
                if id1 not in self._nums:
                    self._nums[id1] = 0
                self._nums[id1] += 1
                for symmelem2 in self._included:
                    id2 = symmelem2.id
                    vec2 = symmelem2.vec
                    angle = intersectangle(vec1, vec2)
                    found = False
                    for special_angle in SPECIAL_ANGLES:
                        diff = abs(angle - special_angle)
                        if diff <= tol:
                            found = True
                            angle = special_angle
                            break
                    if not found:
                        nom, denom = rational(angle / PI, tol)
                        angle = nom * PI / denom
                    if angle == 0.0 and symmelem1.similar(symmelem2):
                        raise ValueError(
                            f"a parallel {symmelem1.name} already included"
                        )
                    key = (id1, id2) if id1 >= id2 else (id2, id1)
                    if key not in self._angles:
                        self._angles[key] = {}
                    if angle not in self._angles[key]:
                        self._angles[key][angle] = 0
                    elif self._angles[key][angle] == 0:
                        raise ValueError(
                            f"the excluded angle of {angle} between"
                            + f" a {symmelem1.name} and a {symmelem2.name}"
                            + " cannot be included"
                        )
                    self._angles[key][angle] += 1
                for symmelem2 in self._excluded:
                    id2 = symmelem2.id
                    vec2 = symmelem2.vec
                    angle = intersectangle(vec1, vec2)
                    if angle > tol:
                        continue
                    angle = 0.0
                    if symmelem1.similar(symmelem2):
                        raise ValueError(
                            f"the excluded parallel {symmelem1.name} cannot"
                            + " be included"
                        )
                    key = (id1, id2) if id1 >= id2 else (id2, id1)
                    if key not in self._angles:
                        self._angles[key] = {}
                    if (
                        angle in self._angles[key]
                        and self._angles[key][angle] > 0
                    ):
                        raise ValueError(
                            f"the included angle of {angle} between"
                            + f" a {symmelem1.name} and a {symmelem2.name}"
                            + " cannot be excluded"
                        )
                    self._angles[key][angle] = 0
                self._included.append(symmelem1)
            elif isinstance(symmelem1, InversionCenter):
                if id1 in self._nums:
                    raise ValueError(
                        f"an {symmelem1.name} already "
                        + ("included" if self._nums[id1] == 1 else "excluded")
                    )
                self._nums[id1] = 1

    def exclude(
        self,
        symmelems: Union[SymmetryElement, Sequence[SymmetryElement]],
        tol: float,
    ) -> None:
        """
        Exclude information of one or multiple symmetry elements `symmelems`
        using a tolerance `tol` to calculate exact intersection angles.
        """
        if not isinstance(symmelems, Sequence):
            symmelems = (symmelems,)
        for symmelem1 in symmelems:
            id1 = symmelem1.id
            if isinstance(symmelem1, _DirectionSymmetryElements):
                vec1 = symmelem1.vec
                for symmelem2 in self._included:
                    id2 = symmelem2.id
                    vec2 = symmelem2.vec
                    angle = intersectangle(vec1, vec2)
                    if angle > tol:
                        continue
                    angle = 0.0
                    if symmelem1.similar(symmelem2):
                        raise ValueError(
                            f"the included parallel {symmelem1.name} cannot"
                            + " be excluded"
                        )
                    key = (id1, id2) if id1 >= id2 else (id2, id1)
                    if key not in self._angles:
                        self._angles[key] = {}
                    if (
                        angle in self._angles[key]
                        and self._angles[key][angle] > 0
                    ):
                        raise ValueError(
                            f"the included angle of {angle} between"
                            + f" a {symmelem1.name} and a {symmelem2.name}"
                            + " cannot be excluded"
                        )
                    self._angles[key][angle] = 0
                for symmelem2 in self._excluded:
                    id2 = symmelem2.id
                    vec2 = symmelem2.vec
                    angle = intersectangle(vec1, vec2)
                    if angle <= tol and symmelem1.similar(symmelem2):
                        raise ValueError(
                            f"a parallel {symmelem1.name} already excluded"
                        )
                self._excluded.append(symmelem1)
            elif isinstance(symmelem1, InversionCenter):
                if id1 in self._nums:
                    raise ValueError(
                        f"an {symmelem1.name} already "
                        + ("included" if self._nums[id1] == 1 else "excluded")
                    )
                self._nums[id1] = 0

    def contains(self, other: "PointGroupInfo") -> bool:
        """
        Check whether another instance `other` is a subset of the instance.
        """
        for key1, num in other.nums.items():
            if key1 in self._nums:
                ref_num = self._nums[key1]
            else:
                ref_num = 0
            zero = num == 0
            ref_zero = ref_num == 0
            if ref_num < num or zero != ref_zero:
                return False
        for key2, angles in other.angles.items():
            for angle, num in angles.items():
                if angle in self._angles[key2]:
                    ref_num = self._angles[key2][angle]
                else:
                    ref_num = 0
                zero = num == 0
                ref_zero = ref_num == 0
                if ref_num < num or zero != ref_zero:
                    return False
        return True


_PointGroup = TypeVar("_PointGroup", bound="PointGroup")


class PointGroup(Transformable):
    """Point group."""

    def __init__(
        self, symb: str, transform: Transformation = Identity()
    ) -> None:
        """
        Initialize the instance with a symbol `symb` and a transformation
        `transform` describing the orientation in space.
        """
        if not symb:
            raise ValueError("empty symbol")
        PRIMAX = vector((0.0, 0.0, 1.0))
        SECAX = vector((1.0, 0.0, 0.0))
        vec: RealVector
        symmelems: List[SymmetryElement] = []
        labels: List[str] = []

        def add(symmelem: SymmetryElement, label: str = "") -> None:
            symmelems.append(symmelem)
            labels.append(label)

        while symb:
            rotation = symb[0]
            subscript = symb[1:]
            if subscript.startswith(SYMB.inf):
                i = len(SYMB.inf)
                order = SYMB.inf
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
                        raise ValueError(
                            "leading zero in the order of the symbol"
                        )
                    n = int(order)
                else:
                    n = 0
                inf = False
                reflection = subscript[i:]
            self._symb = symb
            symb = ""
            if rotation == "C":
                if order:
                    if not reflection:
                        if n > 1:
                            add(RotationAxis(PRIMAX, n))
                        elif inf:
                            add(InfRotationAxis(PRIMAX))
                    elif reflection == "i":
                        if n == 1:
                            symb = "Ci"
                        elif n % 2 == 1:
                            symb = f"S{2 * n}"
                        elif (n // 2) % 2 == 1:
                            symb = f"S{n // 2}"
                        else:
                            symb = f"S{order}"
                    elif reflection == "v":
                        if n == 1:
                            symb = "Cs"
                        else:
                            if not inf:
                                add(RotationAxis(PRIMAX, n))
                                vec = SECAX.copy()
                                step = PI / (2 * n)
                                add(ReflectionPlane(vec), "v")
                                if n % 2 == 1:
                                    angle = step
                                    for _ in range(1, n):
                                        vec[0] = cos(angle)
                                        vec[1] = sin(angle)
                                        add(ReflectionPlane(vec), "v")
                                        angle += step
                                else:
                                    angle = step
                                    step += step
                                    for _ in range(1, n, 2):
                                        vec[0] = cos(angle)
                                        vec[1] = sin(angle)
                                        add(ReflectionPlane(vec), "d")
                                        angle += step
                                    angle = step
                                    for _ in range(2, n, 2):
                                        vec[0] = cos(angle)
                                        vec[1] = sin(angle)
                                        add(ReflectionPlane(vec), "v")
                                        angle += step
                            else:
                                add(InfRotationAxis(PRIMAX))
                                add(AxisReflectionPlanes(PRIMAX), "v")
                    elif reflection == "h":
                        if n == 1:
                            symb = "Cs"
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
                            "a symbol starting with 'C' and an order can end"
                            + " only with '', 'i', 'v', or 'h'"
                        )
                elif reflection == "s":
                    add(ReflectionPlane(PRIMAX))
                elif reflection == "i":
                    add(InversionCenter())
                else:
                    raise ValueError(
                        "a symbol starting with 'C' should have an order or"
                        + " end with 's' or 'i'"
                    )
            elif rotation == "S":
                if reflection:
                    raise ValueError(
                        "a symbol starting with 'S' can end only with an order"
                    )
                if n % 2 == 1 or inf:
                    symb = f"C{order}h"
                elif n == 2:
                    symb = "Ci"
                elif n > 0:
                    add(RotoreflectionAxis(PRIMAX, n))
                    add(RotationAxis(PRIMAX, n // 2))
                    if (n // 2) % 2 == 1:
                        add(InversionCenter())
                else:
                    raise ValueError(
                        "a symbol starting with 'S' should have an order"
                    )
            elif rotation == "D":
                if n > 0:
                    add(RotationAxis(PRIMAX, n))
                    vec = SECAX.copy()
                    step = PI / (2 * n)
                    add(RotationAxis(vec, 2), "'")
                    if n % 2 == 1:
                        angle = step
                        for _ in range(1, n):
                            vec[0] = cos(angle)
                            vec[1] = sin(angle)
                            add(RotationAxis(vec, 2), "'")
                            angle += step
                    else:
                        angle = step
                        step += step
                        for _ in range(1, n, 2):
                            vec[0] = cos(angle)
                            vec[1] = sin(angle)
                            add(RotationAxis(vec, 2), "''")
                            angle += step
                        angle = step
                        for _ in range(2, n, 2):
                            vec[0] = cos(angle)
                            vec[1] = sin(angle)
                            add(RotationAxis(vec, 2), "'")
                            angle += step
                elif inf:
                    add(InfRotationAxis(PRIMAX))
                    add(AxisRotationAxes(PRIMAX))
                else:
                    raise ValueError(
                        "a symbol starting with 'D' should have an order"
                    )
                if not reflection:
                    if n == 1:
                        symb = f"C{2 * n}"
                    elif inf:
                        add(InversionCenter())
                elif reflection == "d":
                    if n == 1:
                        symb = f"C{2 * n}h"
                    elif inf:
                        symb = f"D{order}h"
                    else:
                        vec = SECAX.copy()
                        step = PI / (2 * n)
                        angle = 0.5 * step
                        for _ in range(n):
                            vec[0] = cos(angle)
                            vec[1] = sin(angle)
                            add(ReflectionPlane(vec), "d")
                            angle += step
                        if n % 2 == 1:
                            add(InversionCenter())
                        add(RotoreflectionAxis(PRIMAX, 2 * n))
                elif reflection == "h":
                    if n == 1:
                        symb = f"C{2 * n}v"
                    else:
                        add(ReflectionPlane(PRIMAX), "h")
                        if not inf:
                            vec = SECAX.copy()
                            step = PI / (2 * n)
                            add(ReflectionPlane(vec), "v")
                            if n % 2 == 1:
                                angle = step
                                for _ in range(1, n):
                                    vec[0] = cos(angle)
                                    vec[1] = sin(angle)
                                    add(ReflectionPlane(vec), "v")
                                    angle += step
                            else:
                                angle = step
                                step += step
                                for _ in range(1, n, 2):
                                    vec[0] = cos(angle)
                                    vec[1] = sin(angle)
                                    add(ReflectionPlane(vec), "d")
                                    angle += step
                                angle = step
                                for _ in range(2, n, 2):
                                    vec[0] = cos(angle)
                                    vec[1] = sin(angle)
                                    add(ReflectionPlane(vec), "v")
                                    angle += step
                                add(InversionCenter())
                            if n > 2:
                                add(RotoreflectionAxis(PRIMAX, n))
                        else:
                            add(AxisReflectionPlanes(PRIMAX), "v")
                            add(InversionCenter())
            elif order:
                raise ValueError(
                    "only the symbols starting with 'C', 'S', or 'D' can have"
                    + " an order"
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
                elif reflection:
                    raise ValueError(
                        "a symbol starting with 'T' can end only with '', 'd',"
                        + " or 'h'"
                    )
            elif rotation == "O":
                vecs4 = ax3permut([[1]])
                vecs3 = signvar([1, 1, 1], 1)
                vecs2 = ax3permut(signvar([1, 1], 0, True))
                for n, vecs, label in (
                    (4, vecs4, ""),
                    (3, vecs3, ""),
                    (2, vecs2, "'"),
                ):
                    for vec in vecs:
                        add(RotationAxis(vec, n), label)
                if reflection == "h":
                    for vec in vecs4:
                        add(ReflectionPlane(vec), "h")
                    for vec in vecs2:
                        add(ReflectionPlane(vec), "d")
                    add(InversionCenter())
                    for n, vecs in ((6, vecs3), (4, vecs4)):
                        for vec in vecs:
                            add(RotoreflectionAxis(vec, n))
                elif reflection:
                    raise ValueError(
                        "a symbol starting with 'O' can end only with '' or"
                        + " 'h'"
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
                elif reflection:
                    raise ValueError(
                        "a symbol starting with 'I' can end only with '' or"
                        + " 'h'"
                    )
            elif rotation == "K":
                add(CenterRotationAxes())
                if reflection == "h":
                    add(CenterReflectionPlanes())
                    add(InversionCenter())
                    add(CenterRotoreflectionAxes())
                elif reflection:
                    raise ValueError(
                        "a symbol starting with 'K' can end only with '' or"
                        + " 'h'"
                    )
            else:
                raise ValueError(
                    "a symbol can start only with 'C', 'S', 'D', 'T', 'O',"
                    + " 'I', or 'K'"
                )

        self._symmelems = (
            tuple(symmelems)
            if isinstance(transform, Identity)
            else tuple(transform(symmelem) for symmelem in symmelems)
        )
        self._labels = tuple(labels)
        self._transform = transform

    @property
    def symb(self) -> str:
        """Return the symbol."""
        return self._symb

    @property
    def symmelems(self) -> Sequence[SymmetryElement]:
        """Return the symmetry elements."""
        return self._symmelems

    @property
    def transform(self) -> Transformation:
        """Return the transformation describing the orientation in space."""
        return self._transform

    @property
    def args(self) -> str:
        res = f"'{self._symb}'"
        if not isinstance(self._transform, Identity):
            res += f",{self._transform}"
        return res

    @property
    def props(self) -> Tuple:
        return super().props + (self._symb,)

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, self._transform.diff(obj.transform))
        return res

    def translate(self: _PointGroup, transl: Translation) -> _PointGroup:
        return self.copy()

    def invert(self: _PointGroup) -> _PointGroup:
        res = self.copy()
        res._transform = self._transform.invert()
        return res

    def rotate(self: _PointGroup, rot: Rotation) -> _PointGroup:
        res = self.copy()
        res._transform = self._transform.rotate(rot)
        return res

    def reflect(self: _PointGroup, refl: Reflection) -> _PointGroup:
        res = self.copy()
        res._transform = self._transform.reflect(refl)
        return res

    def rotoreflect(
        self: _PointGroup, rotorefl: Rotoreflection
    ) -> _PointGroup:
        res = self.copy()
        res._transform = self._transform.rotoreflect(rotorefl)
        return res

    @classmethod
    def from_symmelem_nums(
        cls, symmelems: Sequence[SymmetryElement]
    ) -> "PointGroup":
        """
        Construct an instance from a set of symmetry elements `symmelems` using
        only their types and numbers.
        """
        nums = symmelems2nums(symmelems)
        max_rot_order = 0
        max_rotorefl_order = 0
        rot2_num = 0
        refl_num = 0
        invertible = False
        for key, num in nums.items():
            symmelem_type, order = key
            order = abs(order)
            if symmelem_type is RotationAxis:
                if max_rot_order < order:
                    max_rot_order = order
                if order == 2:
                    rot2_num = num
            elif symmelem_type is RotoreflectionAxis:
                if max_rotorefl_order < order:
                    max_rotorefl_order = order
            elif symmelem_type is ReflectionPlane:
                refl_num = num
            elif symmelem_type is InversionCenter:
                invertible = True
        variants: Dict[str, Dict[Tuple[Type[SymmetryElement], int], int]] = {}
        variants.update(_LOW_POINT_GROUP_NUMS)
        new_variants: Set[Tuple[int, int, int]] = set()

        def add(rot: str, order: int, refl: str = "") -> None:
            new_variants.add(
                (ROT_SYMBS.index(rot), order, REFL_SYMBS.index(refl))
            )

        if max_rot_order > 1:
            n = max_rot_order
            add("C", n)
            add("C", n, "v")
            add("C", n, "h")
            add("S", 2 * n)
            add("D", n)
            add("D", n, "d")
            add("D", n, "h")
        if max_rotorefl_order > 2:
            n = max_rotorefl_order
            add("C", n, "h")
            if n % 2 == 0:
                add("S", n)
                add("D", n // 2, "d")
            add("D", n, "h")
        if rot2_num > 1:
            n = rot2_num
            n1 = n
            n2 = n
            if invertible:
                if n % 2 == 0:
                    n1 += 1
                else:
                    n2 += 1
            add("D", n)
            add("D", n1, "d")
            add("D", n2, "h")
        if refl_num > 1:
            n = refl_num
            n1 = n
            n2 = n - 1
            if invertible and n % 2 == 0:
                n1 += 1
                n2 += 1
            add("C", n, "v")
            add("D", n1, "d")
            add("D", n2, "h")
        for rot, order, refl in sorted(new_variants):
            variant = f"{ROT_SYMBS[rot]}{order}{REFL_SYMBS[refl]}".strip()
            variants[variant] = symmelems2nums(PointGroup(variant).symmelems)
        variants.update(_HIGH_POINT_GROUP_NUMS)
        remove = []
        for variant, ref_nums in variants.items():
            for key, num in nums.items():
                if key not in ref_nums or ref_nums[key] < num:
                    remove.append(variant)
                    break
        for variant in remove:
            del variants[variant]
        keys = tuple(variants.keys())
        if len(keys) == 0:
            raise ValueError("invalid combination of symmetry elements")
        return cls(keys[0])


_LOW_POINT_GROUPS = (
    "C1",
    "Cs",
    "Ci",
)
_HIGH_POINT_GROUPS = (
    "T",
    "Td",
    "Th",
    "O",
    "Oh",
    "I",
    "Ih",
    f"C{SYMB.inf}",
    f"C{SYMB.inf}v",
    f"C{SYMB.inf}h",
    f"D{SYMB.inf}",
    f"D{SYMB.inf}h",
    "K",
    "Kh",
)
_LOW_POINT_GROUP_NUMS = {
    symb: symmelems2nums(PointGroup(symb).symmelems)
    for symb in _LOW_POINT_GROUPS
}
_HIGH_POINT_GROUP_NUMS = {
    symb: symmelems2nums(PointGroup(symb).symmelems)
    for symb in _HIGH_POINT_GROUPS
}
