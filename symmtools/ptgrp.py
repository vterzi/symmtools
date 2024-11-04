"""Class for point groups."""

__all__ = ["PointGroup"]

from numpy.linalg import eigh

from .const import (
    INF,
    PI,
    TOL,
    ORIGIN,
    PRIMAX,
    SECAX,
    SYMB,
    ROT_SYMBS,
    REFL_SYMBS,
)
from .vecop import (
    vector,
    norm,
    normalize,
    orthogonalize,
    cross,
    zero,
    parallel,
    unitparallel,
    perpendicular,
    angle,
    inertia,
)
from .transform import (
    Transformable,
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
from .quaternion import Quaternion
from .ptgrpinfo import PointGroupInfo, PointGroupInfos, VARIANTS
from .primitive import Points
from .typehints import (
    TypeVar,
    Union,
    Optional,
    Any,
    Sequence,
    Iterator,
    Set,
    FrozenSet,
    Tuple,
    List,
    Vector,
    Dict,
)

_ORIGIN = vector(ORIGIN)
_PRIMAX = vector(PRIMAX)
_SECAX = vector(SECAX)


def axis_transform(
    from_axis: Vector, to_axis: Vector
) -> Union[Identity, Rotation]:
    """
    Return the (proper) transformation from one axis `from_axis` to another
    axis `to_axis`.  Antiparallel axis vectors are considered to be equivalent.
    """
    axis = cross(from_axis, to_axis)
    return (
        Rotation(axis, angle(from_axis, to_axis))
        if axis.dot(axis) > 0.0
        else Identity()
    )


def axes_transform(
    from_axis1: Vector, from_axis2: Vector, to_axis1: Vector, to_axis2: Vector
) -> Union[Identity, Rotation]:
    """
    Return the (proper) transformation from one pair of axes `from_axis1` and
    `from_axis2` to another pair of axes `to_axis1` and `to_axis2`.  The axes
    in each pair are assumed to be orthogonal to each other.
    """
    transform1 = axis_transform(from_axis1, to_axis1)
    if isinstance(transform1, Identity) and from_axis1.dot(to_axis1) < 0.0:
        transform1 = Rotation(to_axis2, PI)
    from_axis2 = transform1.apply(from_axis2)
    transform2 = axis_transform(from_axis2, to_axis2)
    if isinstance(transform2, Identity) and from_axis2.dot(to_axis2) < 0.0:
        transform2 = Rotation(to_axis1, PI)
    if isinstance(transform2, Rotation):
        if isinstance(transform1, Rotation):
            return (
                Quaternion.from_rotation(transform2)
                * Quaternion.from_rotation(transform1)
            ).rotation
        else:
            return transform2
    else:
        return transform1


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
        self, symb: str, transform: Union[Identity, Rotation] = Identity()
    ) -> None:
        """
        Initialize the instance with a symbol `symb` and a transformation
        `transform` describing the orientation in space.
        """
        if not symb:
            raise ValueError("empty symbol")

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
                    raise ValueError("leading zero in the order of the symbol")
                n = int(order)
            else:
                n = 0
            inf = False
            reflection = subscript[i:]
        info: PointGroupInfo
        if rotation == "C":
            if order:
                if not reflection:
                    if inf:
                        info = PointGroupInfos.Coo
                    elif n == 1:
                        info = PointGroupInfos.C1
                    else:
                        info = PointGroupInfos.Cn(n)
                elif reflection == "i":
                    if inf:
                        symb = f"C{order}h"
                        info = PointGroupInfos.Cooh
                    elif n == 1:
                        symb = "Ci"
                        info = PointGroupInfos.Ci
                    elif n == 2:
                        symb = "Cs"
                        info = PointGroupInfos.Cs
                    elif n % 4 == 0:
                        symb = f"S{n}"
                        info = PointGroupInfos.S2n(n // 2)
                    elif n % 2 == 1:
                        symb = f"S{2 * n}"
                        info = PointGroupInfos.S2n(n)
                    else:
                        n //= 2
                        symb = f"C{n}h"
                        info = PointGroupInfos.Cnh(n)
                elif reflection == "v":
                    if inf:
                        info = PointGroupInfos.Coov
                    elif n == 1:
                        symb = "Cs"
                        info = PointGroupInfos.Cs
                    else:
                        info = PointGroupInfos.Cnv(n)
                elif reflection == "h":
                    if inf:
                        info = PointGroupInfos.Cooh
                    elif n == 1:
                        symb = "Cs"
                        info = PointGroupInfos.Cs
                    else:
                        info = PointGroupInfos.Cnh(n)
                else:
                    raise ValueError(
                        "a symbol starting with 'C' and an order can end only"
                        + " with '', 'i', 'v', or 'h'"
                    )
            elif reflection == "s":
                info = PointGroupInfos.Cs
            elif reflection == "i":
                info = PointGroupInfos.Ci
            else:
                raise ValueError(
                    "a symbol starting with 'C' should have an order or end"
                    + " with 's' or 'i'"
                )
        elif rotation == "S":
            if reflection:
                raise ValueError(
                    "a symbol starting with 'S' can end only with an order"
                )
            if order:
                if inf:
                    symb = f"C{order}h"
                    info = PointGroupInfos.Cooh
                elif n == 1:
                    symb = "Cs"
                    info = PointGroupInfos.Cs
                elif n == 2:
                    symb = "Ci"
                    info = PointGroupInfos.Ci
                elif n % 2 == 1:
                    symb = f"C{n}h"
                    info = PointGroupInfos.Cnh(n)
                else:
                    info = PointGroupInfos.S2n(n // 2)
            else:
                raise ValueError(
                    "a symbol starting with 'S' should have an order"
                )
        elif rotation == "D":
            if order:
                if not reflection:
                    if inf:
                        info = PointGroupInfos.Doo
                    elif n == 1:
                        symb = "C2"
                        info = PointGroupInfos.Cn(2)
                    else:
                        info = PointGroupInfos.Dn(n)
                elif reflection == "d":
                    if inf:
                        symb = f"D{order}h"
                        info = PointGroupInfos.Dooh
                    elif n == 1:
                        symb = "C2h"
                        info = PointGroupInfos.Cnh(2)
                    else:
                        info = PointGroupInfos.Dnd(n)
                elif reflection == "h":
                    if inf:
                        info = PointGroupInfos.Dooh
                    elif n == 1:
                        symb = "C2v"
                        info = PointGroupInfos.Cnv(2)
                    else:
                        info = PointGroupInfos.Dnh(n)
            else:
                raise ValueError(
                    "a symbol starting with 'D' should have an order"
                )
        elif order:
            raise ValueError(
                "only the symbols starting with 'C', 'S', or 'D' can have an"
                + " order"
            )
        elif rotation == "T":
            if not reflection:
                info = PointGroupInfos.T
            if reflection == "d":
                info = PointGroupInfos.Td
            elif reflection == "h":
                info = PointGroupInfos.Th
            elif reflection:
                raise ValueError(
                    "a symbol starting with 'T' can end only with '', 'd', or"
                    + " 'h'"
                )
        elif rotation == "O":
            if not reflection:
                info = PointGroupInfos.O
            elif reflection == "h":
                info = PointGroupInfos.Oh
            elif reflection:
                raise ValueError(
                    "a symbol starting with 'O' can end only with '' or 'h'"
                )
        elif rotation == "I":
            if not reflection:
                info = PointGroupInfos.I
            elif reflection == "h":
                info = PointGroupInfos.Ih
            elif reflection:
                raise ValueError(
                    "a symbol starting with 'I' can end only with '' or 'h'"
                )
        elif rotation == "K":
            if not reflection:
                info = PointGroupInfos.K
            if reflection == "h":
                info = PointGroupInfos.Kh
            elif reflection:
                raise ValueError(
                    "a symbol starting with 'K' can end only with '' or 'h'"
                )
        else:
            raise ValueError(
                "a symbol can start only with 'C', 'S', 'D', 'T', 'O', 'I', or"
                + " 'K'"
            )

        self._symb = symb
        self._info = info
        self._transform = transform

    @property
    def symb(self) -> str:
        """Return the symbol."""
        return self._symb

    @property
    def symm_elems(self) -> Iterator[SymmetryElement]:
        """Return the symmetry elements."""
        for symm_elem in self._info.symm_elems:
            yield self._transform(symm_elem)

    @property
    def types(self) -> Dict[Tuple, int]:
        """Return the types of symmetry elements and their numbers."""
        return self._info.types

    @property
    def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
        """
        Return the angles between axes or normals of symmetry elements and
        their numbers.
        """
        return self._info.angles

    @property
    def transform(self) -> Union[Identity, Rotation]:
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
        if not points.nondegen(tol):
            raise ValueError(
                "at least two identical elements in the set of points for the"
                + " given tolerance"
            )
        points = points.center()
        poses = tuple(elem.pos for elem in points._elems)

        invertible = InversionCenter().symmetric(points, tol)
        n_points = len(poses)

        if n_points == 1:
            # degenerate: Kh
            return cls("Kh")

        def transformation(
            axis1: Vector, axis2: Optional[Vector] = None
        ) -> Union[Identity, Rotation]:
            if axis2:
                return axes_transform(_PRIMAX, _SECAX, axis1, axis2)
            else:
                return axis_transform(_PRIMAX, axis1)

        main_axis = _ORIGIN
        for i in range(n_points):
            pos = poses[i]
            if not zero(pos, tol):
                main_axis = pos
                break
        for i in range(i + 1, n_points):
            if not parallel(main_axis, poses[i], tol):
                break
        else:
            # linear: Coov,Dooh
            if _PRIMAX.dot(main_axis) < 0.0:
                main_axis = -main_axis
            return cls(
                "Dooh" if invertible else "Coov", transformation(main_axis)
            )

        axes: List[Vector] = []
        normals: List[Vector] = []

        def new(arr: List[Vector], vec: Vector) -> bool:  # TODO remove
            for elem in arr:
                if unitparallel(elem, vec, tol):
                    return False
            arr.append(vec)
            return True

        eigvals, eigvecs = eigh(inertia(poses))
        oblate = eigvals[1] - eigvals[0] <= tol
        prolate = eigvals[2] - eigvals[1] <= tol
        if oblate and prolate:
            # spherical: T,Td,Th,O,Oh,I,Ih
            rots: List[RotationAxis] = []
            complete = False
            for _, idxs in points._groups:
                collinear_part = False
                coplanar_part = False
                for i1 in range(n_points - 2):
                    pos1 = poses[idxs[i1]]
                    for i2 in range(i1 + 1, n_points - 1):
                        pos2 = poses[idxs[i2]]
                        segment = pos1 - pos2
                        if i2 == 1:
                            collinear_part = True
                        for i3 in range(i2 + 1, n_points):
                            pos3 = poses[idxs[i3]]
                            normal = cross(segment, pos1 - pos3)
                            normal_norm = norm(normal)
                            if normal_norm <= tol:
                                continue
                            axis = normal / normal_norm
                            collinear_part = False
                            if new(axes, axis):
                                dist = pos1.dot(axis)
                                max_order = 3
                                for i4 in range(n_points):
                                    if i4 == i1 or i4 == i2 or i4 == i3:
                                        continue
                                    pos4 = poses[idxs[i4]]
                                    if abs(pos4.dot(axis) - dist) <= tol:
                                        max_order += 1
                                if i3 == 2 and max_order == n_points:
                                    coplanar_part = True
                                for order in range(max_order, 1, -1):
                                    if (
                                        max_order % order != 0
                                        and (max_order - 1) % order != 0
                                    ):
                                        continue
                                    rot = RotationAxis(axis, order)
                                    if rot.symmetric(points, tol):
                                        rots.append(rot)
                                        if len(rots) == 2:
                                            complete = True
                                        break
                        if collinear_part or coplanar_part or complete:
                            break
                    else:
                        continue
                    break
                if complete:
                    break
            else:
                raise ValueError("no two rotation axes found")
            rot1, rot2 = rots
            vec1 = rot1.vec
            vec2 = rot2.vec
            order1 = rot1.order
            order2 = rot2.order
            if order1 < order2:
                vec1, vec2 = vec2, vec1
                order1, order2 = order2, order1
            if vec1.dot(vec2) < 0.0:
                vec2 = -vec2
            original_axes = [
                vec1,
                normalize(orthogonalize(vec2, vec1)),
            ]
            original_axes.append(
                normalize(cross(original_axes[0], original_axes[1]))
            )
            ang = angle(vec1, vec2)
            min_diff = INF
            for ref_ang, ref_variants in VARIANTS[(order1, order2)].items():
                diff = abs(ang - ref_ang)
                if min_diff > diff:
                    min_diff = diff
                    variants = ref_variants
            suffix = "h" if invertible else ""
            n_variants = len(variants)
            for i_variant in range(n_variants):
                vecs_obj, axes_order = variants[i_variant]
                permut_axes = tuple(
                    (
                        original_axes[i_axis]
                        if i_axis >= 0
                        else -original_axes[-i_axis]
                    )
                    for i_axis in axes_order
                )
                if suffix == "" and vecs_obj.symb == "T":
                    components = vecs_obj.vecs[1][0]
                    rotorefl = RotoreflectionAxis(
                        components[0] * permut_axes[0]
                        + components[1] * permut_axes[1]
                        + components[2] * permut_axes[2],
                        4,
                    )
                    if rotorefl.symmetric(points, tol):
                        suffix = "d"
                if i_variant < n_variants - 1:
                    generator = vecs_obj.symm_elems(
                        permut_axes[0],
                        permut_axes[1],
                        permut_axes[2],
                        suffix,
                    )
                    symm_elem = next(generator)
                    if symm_elem.symmetric(points, tol):
                        break
            return cls(
                f"{vecs_obj.symb}{suffix}",
                transformation(permut_axes[0], permut_axes[1]),
            )
        elif oblate or prolate:
            # symmetric: C(n>2),C(n>2)v,C(n>2)h,S(2n),D(n>3),Dnd,D(n>3)h
            main_axis = eigvecs[:, 2] if oblate else eigvecs[:, 0]
            axes.append(main_axis)
            coplanar = True
            orders = set()
            for _, idxs in points._groups:
                dists: Dict[float, int] = {}
                n_points = len(idxs)
                for i in range(n_points):
                    pos = poses[idxs[i]]
                    dist = pos.dot(main_axis)
                    if coplanar and dist > tol:
                        coplanar = False
                    for ref_dist in dists:
                        if abs(dist - ref_dist) <= tol:
                            dists[ref_dist] += 1
                            break
                    else:
                        dists[dist] = 1
                for count in dists.values():
                    orders.add(count)
            ref_orders = sorted(orders, reverse=True)
            for order in range(ref_orders[0], 1, -1):
                for ref_order in ref_orders:
                    if ref_order % order != 0 and (ref_order - 1) % order != 0:
                        break
                else:
                    if RotationAxis(main_axis, order).symmetric(points, tol):
                        max_order = order
                        rotorefl_factor = 0
                        if not coplanar:
                            for factor in (2, 1):
                                new_order = order * factor
                                if new_order > 2:
                                    rotorefl = RotoreflectionAxis(
                                        main_axis, new_order
                                    )
                                    if rotorefl.symmetric(points, tol):
                                        rotorefl_factor = factor
                                        break
                        elif order > 2:
                            rotorefl_factor = 1
                        break
            for _, idxs in points._groups:
                n_points = len(idxs)
                for i1 in range(n_points - 1):
                    pos1 = poses[idxs[i1]]
                    for i2 in range(i1 + 1, n_points):
                        pos2 = poses[idxs[i2]]
                        segment = pos1 - pos2
                        midpoint = 0.5 * (pos1 + pos2)
                        if not perpendicular(segment, midpoint, tol):
                            continue
                        midpoint_norm = norm(midpoint)
                        nonzero = midpoint_norm > tol
                        if nonzero or coplanar:
                            axis = (
                                midpoint / midpoint_norm
                                if nonzero
                                else normalize(cross(segment, main_axis))
                            )
                            if (
                                (perpendicular(axis, main_axis, tol))
                                and new(axes, axis)
                                and RotationAxis(axis, 2).symmetric(
                                    points, tol
                                )
                            ):
                                transform = transformation(main_axis, axis)
                                if rotorefl_factor == 2:
                                    return cls(f"D{max_order}d", transform)
                                elif rotorefl_factor == 1:
                                    return cls(f"D{max_order}h", transform)
                                else:
                                    return cls(f"D{max_order}", transform)
                        normal = normalize(segment)
                        if (
                            (perpendicular(normal, main_axis, tol))
                            and new(normals, normal)
                            and ReflectionPlane(normal).symmetric(points, tol)
                        ):
                            transform = transformation(
                                main_axis,
                                (
                                    Rotation(
                                        main_axis, PI / (2 * max_order)
                                    ).apply(normal)
                                    if rotorefl_factor == 2
                                    else normal
                                ),
                            )
                            if rotorefl_factor == 2:
                                return cls(f"D{max_order}d", transform)
                            elif rotorefl_factor == 1:
                                return cls(f"D{max_order}h", transform)
                            else:
                                return cls(f"C{max_order}v", transform)
                transform = transformation(main_axis)
                if rotorefl_factor == 2:
                    return cls(f"S{2*max_order}", transform)
                elif rotorefl_factor == 1:
                    return cls(f"C{max_order}h", transform)
                else:
                    return cls(f"C{max_order}", transform)
        else:
            # asymmetric: C1,Cs,Ci,C2,C2v,C2h,D2,D2h
            nums = 4 * [0]
            axes = []
            for i in range(3):
                vec = eigvecs[:, i]
                found_rot = RotationAxis(vec, 2).symmetric(points, tol)
                if invertible:
                    found_refl = found_rot
                else:
                    found_refl = ReflectionPlane(vec).symmetric(points, tol)
                if found_rot or found_refl:
                    axes.append(vec)
                idx = 0
                if found_rot:
                    idx += 2
                if found_refl:
                    idx += 1
                nums[idx] += 1
                if i > 0:
                    if len(axes) == 2:
                        transform = transformation(axes[0], axes[1])
                    elif len(axes) == 1:
                        transform = transformation(axes[0])
                    if invertible:
                        if nums[3] > 1:
                            return cls("D2h", transform)
                        elif nums[3] == 1:
                            return cls("C2h", transform)
                        else:
                            return cls("Ci")
                    else:
                        if nums[2] > 1:
                            return cls("D2", transform)
                        elif nums[2] == 1 and nums[1] > 0:
                            return cls("C2v", transform)
                        elif nums[2] == 1:
                            return cls("C2", transform)
                        elif nums[1] == 1:
                            return cls("Cs", transform)
                        else:
                            return cls("C1")
        return cls("C1")

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
        for prop, num in info.types:
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
            ref_info.include(tuple(group.symm_elems), TOL)
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
        info.include(tuple(group.symm_elems), TOL)
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
