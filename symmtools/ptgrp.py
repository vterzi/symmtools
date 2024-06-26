"""Class for point groups."""

__all__ = ["PointGroup"]

from math import sin, cos

from .const import (
    INF,
    PI,
    PHI,
    TOL,
    SYMB,
    ROT_SYMBS,
    REFL_SYMBS,
)
from .vecop import vector
from .tools import signvar, ax3permut
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
    SymmetryElements,
)
from .primitive import Points
from .typehints import (
    TypeVar,
    Any,
    Sequence,
    Set,
    Tuple,
    List,
    RealVector,
)


_PointGroup = TypeVar("_PointGroup", bound="PointGroup")


class PointGroup(Transformable):
    """
    Point group.

    Possible point group symbols and their symmetry elements (except E):
    | Symbol       | Rotation      | Reflection | Inversion | Rotoreflection |
    |--------------|---------------|------------|-----------|----------------|
    | C1           |               |            |           |                |
    | Cs           |               | s          |           |                |
    | Ci           |               |            | i         |                |
    | Cn           | Cn            |            |           |                |
    | Cnv          | Cn            | ns         |           |                |
    | C(n=2k)h     | Cn            | s          | i         | Sn             |
    | C(n=2k+1)h   | Cn            | s          |           | Sn             |
    | S(2(n=2k))   | Cn            |            |           | S(2n)          |
    | S(2(n=2k+1)) | Cn            |            | i         | S(2n)          |
    | Dn           | Cn,nC2        |            |           |                |
    | D(n=2k)d     | Cn,nC2        | ns         |           | S(2n)          |
    | D(n=2k+1)d   | Cn,nC2        | ns         | i         | S(2n)          |
    | D(n=2k)h     | Cn,nC2        | (n+1)s     | i         | Sn             |
    | D(n=2k+1)h   | Cn,nC2        | (n+1)s     |           | Sn             |
    | T            | 4C3,3C2       |            |           |                |
    | Td           | 4C3,3C2       | 6s         |           | 3S4            |
    | Th           | 4C3,3C2       | 3s         | i         | 4S6            |
    | O            | 3C4,4C3,6C2   |            |           |                |
    | Oh           | 3C4,4C3,6C2   | 9s         | i         | 4S6,3S4        |
    | I            | 6C5,10C3,15C2 |            |           |                |
    | Ih           | 6C5,10C3,15C2 | 15s        | i         | 6S10,10S6      |
    | Coo          | Coo           |            |           |                |
    | Coov         | Coo           | oosv       |           |                |
    | Cooh         | Coo           | s          | i         | Soo            |
    | Doo          | Coo,ooC2      |            |           |                |
    | Dooh         | Coo,ooC2      | oosv,s     | i         | Soo            |
    | K            | ooCoo         |            |           |                |
    | Kh           | ooCoo         | oos        | i         | ooSoo          |
    General notation:
    - n, k: positive integer
    - oo: infinity
    Symmetry element notation:
    - E: identity element
    - Cn: n-fold rotation axis (high-order if n > 2) (C1 = E)
    - s: reflection plane
    - i: inversion center
    - Sn: n-fold rotoreflection axis (S1 = s, S2 = i)
    - Cni: n-fold rotoinversion axis (C1i = i, C2i = s, C(n=4k)i = Sn,
      C(n=2k+1)i = S(2n), C(2(n=2k+1))i = Sn)
    Point group symbol notation:
    1. Rotation symmetry:
      - C: cyclic
      - S: improper cyclic (Spiegel)
      - D: dihedral
      - T: tetrahedral
      - O: octahedral
      - I: icosahedral
      - K: spherical (Kugel)
    2. Highest (primary) rotation axis order for C (except Cs), S, and D
    3. Reflection symmetry:
    - s: only one reflection plane (Spiegel)
    - i: only one inversion center
    - v: vertical reflection planes (containing the primary rotation axis)
    - d: diagonal reflection planes (containing the primary rotation axis and
      diagonal to the secondary 2-fold rotation axes)
    - h: horizontal reflection planes (perpendicular to high-order rotation
      axes)

    Redundant point group symbols:
    - C1i = Ci
    - C2i = Cs
    - C(n=4k)i = Sn
    - C(n=2k+1)i = S(2n)
    - C(2(n=2k+1))i = Cnh
    - Cooi = Cooh
    - C1v = Cs
    - C1h = Cs
    - S1 = Cs
    - S2 = Ci
    - S(n=2k+1) = Cnh
    - Soo = Cooh
    - D1 = C2
    - D1d = C2h
    - Dood = Dooh
    - D1h = C2v

    Possible point group symbols for different types of rotors:
    | Top        | Point group                                     |
    |------------|-------------------------------------------------|
    | Spherical  | T,Td,Th,O,Oh,I,Ih                               |
    | Degenerate | Kh                                              |
    | Symmetric  | C(n>2),C(n>2)v,C(n>2)h,S(2n),D(n>3),Dnd,D(n>3)h |
    | Linear     | Coov,Dooh                                       |
    | Asymmetric | C1,Cs,Ci,C2,C2v,C2h,D2,D2h                      |
    The relationship between the principal moments of inertia (I1, I2, I3) is:
    - spherical: I1 = I2 = I3
    - degenerate (spherical): 0 = I1 = I2 = I3
    - symmetric: oblate I1 = I2 < I3 or prolate I1 < I2 = I3
    - linear (prolate): 0 = I1 < I2 = I3
    - asymmetric: I1 < I2 < I3
    """

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
        symm_elems: List[SymmetryElement] = []
        labels: List[str] = []

        def add(symm_elem: SymmetryElement, label: str = "") -> None:
            symm_elems.append(symm_elem)
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
                                step = PI / n
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
                    step = PI / n
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
                        step = PI / n
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
                            step = PI / n
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

        self._symm_elems = (
            tuple(symm_elems)
            if isinstance(transform, Identity)
            else tuple(transform(symm_elem) for symm_elem in symm_elems)
        )
        self._labels = tuple(labels)
        self._transform = transform

    @property
    def symb(self) -> str:
        """Return the symbol."""
        return self._symb

    @property
    def symm_elems(self) -> Sequence[SymmetryElement]:
        """Return the symmetry elements."""
        return self._symm_elems

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
    def from_all_symm_elems(
        cls, symm_elems: Sequence[SymmetryElement]
    ) -> "PointGroup":
        """
        Construct an instance from a set of symmetry elements `symm_elems`
        assuming the set is complete.
        """
        rot = ""
        order = ""
        refl = ""
        rot_num = 0
        high_rot_num = 0
        max_rot_order = 1
        refl_num = 0
        invertible = False
        rotorefl_num = 0
        for symm_elem in symm_elems:
            if isinstance(symm_elem, RotationAxis):
                rot_order = symm_elem.order
                rot_num += 1
                if rot_order > 2:
                    high_rot_num += 1
                if max_rot_order < rot_order:
                    max_rot_order = rot_order
            elif isinstance(symm_elem, ReflectionPlane):
                refl_num += 1
            elif isinstance(symm_elem, InversionCenter):
                invertible = True
            elif isinstance(symm_elem, RotoreflectionAxis):
                rotorefl_num += 1
            elif isinstance(symm_elem, InfRotationAxis):
                order = SYMB.inf
            elif isinstance(symm_elem, InfRotoreflectionAxis):
                order = SYMB.inf
                refl = "h"
                if rot:
                    break
                refl_num += 1
                invertible = True
            elif isinstance(symm_elem, AxisRotationAxes):
                rot = "D"
                order = SYMB.inf
                if refl:
                    break
            elif isinstance(symm_elem, AxisReflectionPlanes):
                order = SYMB.inf
                refl_num += 1
            elif isinstance(symm_elem, CenterRotationAxes):
                rot = "K"
            elif isinstance(
                symm_elem, (CenterReflectionPlanes, CenterRotoreflectionAxes)
            ):
                rot = "K"
                refl = "h"
                break
        if not rot:
            if order == SYMB.inf:
                rot = "C"
                if invertible:
                    refl = "h"
                elif refl_num > 0:
                    refl = "v"
            elif high_rot_num > 1:
                if max_rot_order == 5:
                    rot = "I"
                elif max_rot_order == 4:
                    rot = "O"
                else:
                    rot = "T"
                    if refl_num > 0:
                        refl = "d"
                if invertible:
                    refl = "h"
            elif rot_num > 1:
                rot = "D"
                order = str(max_rot_order)
                if refl_num > max_rot_order:
                    refl = "h"
                elif refl_num == max_rot_order:
                    refl = "d"
            elif rotorefl_num > 0 and refl_num == 0:
                rot = "S"
                order = str(2 * max_rot_order)
            else:
                rot = "C"
                if max_rot_order > 1:
                    order = str(max_rot_order)
                    if refl_num == 1:
                        refl = "h"
                    elif refl_num > 1:
                        refl = "v"
                else:
                    if invertible:
                        refl = "i"
                    elif refl_num > 0:
                        refl = "s"
                    else:
                        order = "1"
        return cls(rot + order + refl)

    @classmethod
    def from_points(cls, points: Points, tol: float) -> "PointGroup":
        """
        Construct an instance from a set of points `points` using a tolerance
        `tol` to determine its symmetry elements.
        """
        return cls.from_all_symm_elems(tuple(points.symm_elems(tol)))

    @classmethod
    def from_part_symm_elems(
        cls, symm_elems: Sequence[SymmetryElement], tol: float
    ) -> "PointGroup":
        """
        Construct an instance from a set of symmetry elements `symm_elems`
        using only their types and numbers.
        """
        info = SymmetryElements()
        info.include(symm_elems, tol)
        max_rot_order = 0
        max_rotorefl_order = 0
        rot2_num = 0
        refl_num = 0
        invertible = False
        for prop, num in info.nums:
            symm_elem_type = prop[0]
            if symm_elem_type is RotationAxis:
                order = prop[1]
                if max_rot_order < order:
                    max_rot_order = order
                if order == 2:
                    rot2_num = num
            elif symm_elem_type is RotoreflectionAxis:
                order = prop[1]
                if max_rotorefl_order < order:
                    max_rotorefl_order = order
            elif symm_elem_type is ReflectionPlane:
                refl_num = num
            elif symm_elem_type is InversionCenter:
                invertible = True
        variants = list(_LOW_POINT_GROUPS)
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
        # TODO analyze angles in `info` to suggest point groups
        for rot, order, refl in sorted(new_variants):
            symb = f"{ROT_SYMBS[rot]}{order}{REFL_SYMBS[refl]}".strip()
            group = cls(symb)
            ref_info = SymmetryElements()
            ref_info.include(group.symm_elems, TOL)
            variants.append((group, ref_info))
        variants.extend(_HIGH_POINT_GROUPS)
        remove = []
        for i, elem in enumerate(variants):
            if not elem[1].contains(info):
                remove.append(i)
        for i in reversed(remove):
            del variants[i]
        if len(variants) == 0:
            raise ValueError("invalid combination of symmetry elements")
        return variants[0][0]


def _init(
    symbs: Sequence[str],
) -> Sequence[Tuple[PointGroup, SymmetryElements]]:
    res = []
    for symb in symbs:
        group = PointGroup(symb)
        info = SymmetryElements()
        info.include(group.symm_elems, TOL)
        res.append((group, info))
    return tuple(res)


_LOW_POINT_GROUPS = _init(
    (
        "C1",
        "Cs",
        "Ci",
    )
)
_HIGH_POINT_GROUPS = _init(
    (
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
)
