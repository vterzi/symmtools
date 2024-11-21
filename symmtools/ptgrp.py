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
    Symb,
)
from .utils import (
    vector,
    sqnorm,
    norm,
    normalize,
    orthogonalize,
    cross,
    zero,
    parallel,
    unitparallel,
    perpendicular,
    orthvec,
    angle,
    intersectangle,
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


def vec_rot(
    from_vec: Vector, to_vec: Vector, orth_axis: Optional[Vector] = None
) -> Union[Identity, Rotation]:
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
            return Rotation(axis, ang)
    elif from_vec.dot(to_vec) < 0.0:
        if orth_axis is None:
            orth_axis = orthvec(normalize(to_vec))
        return Rotation(orth_axis, PI)
    return Identity()


def vecs_rot(
    from_vec1: Vector, from_vec2: Vector, to_vec1: Vector, to_vec2: Vector
) -> Union[Identity, Rotation]:
    """
    Return the proper transformation from one pair of orthogonal vectors
    `from_vec1` and `from_vec2` to another pair of orthogonal vectors `to_vec1`
    and `to_vec2`.
    """
    transform1 = vec_rot(from_vec1, to_vec1, to_vec2)
    from_vec2 = transform1.apply(from_vec2)
    transform2 = vec_rot(from_vec2, to_vec2, to_vec1)
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
        if subscript.startswith(Symb.INF):
            i = len(Symb.INF)
            order = Symb.INF
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
        """Symbol."""
        return self._symb

    @property
    def symm_elems(self) -> Iterator[SymmetryElement]:
        """Symmetry elements."""
        for symm_elem in self._info.symm_elems:
            yield self._transform(symm_elem)

    @property
    def types(self) -> Dict[Tuple, int]:
        """Types of symmetry elements and their numbers."""
        return self._info.types

    @property
    def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
        """
        Angles between axes or normals of symmetry elements and their numbers.
        """
        return self._info.angles

    @property
    def transform(self) -> Union[Identity, Rotation]:
        """Transformation describing the orientation in space."""
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
        transform: Union[Identity, Rotation] = Identity()
        invertible = False
        rotorefl_num = 0
        axes: List[Tuple[int, Vector]] = []
        normals: List[Vector] = []
        pot_main_axis = _ORIGIN
        for symm_elem in symm_elems:
            if isinstance(symm_elem, RotationAxis):
                rot_order = symm_elem.order
                i = 0
                n = len(axes)
                while i < n and axes[i][0] > rot_order:
                    i += 1
                axes.insert(i, (rot_order, symm_elem.vec))
            elif isinstance(symm_elem, ReflectionPlane):
                normals.append(symm_elem.vec)
            elif isinstance(symm_elem, InversionCenter):
                invertible = True
            elif isinstance(symm_elem, RotoreflectionAxis):
                rotorefl_num += 1
                pot_main_axis = symm_elem.vec
            elif isinstance(symm_elem, InfRotationAxis):
                order = Symb.INF
                axes.insert(0, (0, symm_elem.vec))
            elif isinstance(symm_elem, InfRotoreflectionAxis):
                order = Symb.INF
                refl = "h"
                if rot:
                    transform = vec_rot(_PRIMAX, symm_elem.vec)
                    break
                invertible = True
            elif isinstance(symm_elem, AxisRotationAxes):
                rot = "D"
                order = Symb.INF
                if refl:
                    transform = vec_rot(_PRIMAX, symm_elem.vec)
                    break
            elif isinstance(symm_elem, AxisReflectionPlanes):
                order = Symb.INF
                # Adding a reflection plane is necessary to differentiate
                # "Coov" from "Coo".
                normals.append(symm_elem.vec)
            elif isinstance(symm_elem, CenterRotationAxes):
                rot = "K"
            elif isinstance(
                symm_elem, (CenterReflectionPlanes, CenterRotoreflectionAxes)
            ):
                rot = "K"
                refl = "h"
                break
        if not rot:
            rot_num = len(axes)
            max_rot_order = axes[0][0] if rot_num > 0 else 1
            high_rot_num = 0
            for i in range(rot_num):
                if axes[i][0] < 3:
                    break
                high_rot_num += 1
            refl_num = len(normals)
            if order == Symb.INF:
                rot = "C"
                if invertible:
                    refl = "h"
                elif refl_num > 0:
                    refl = "v"
                transform = vec_rot(_PRIMAX, axes[0][1])
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
                vec1 = axes[0][1]
                vec2 = axes[1][1]
                if vec1.dot(vec2) < 0.0:
                    vec2 = -vec2
                vec2 = normalize(orthogonalize(vec2, vec1))
                transform = vecs_rot(_PRIMAX, _SECAX, vec1, vec2)
            elif rot_num > 1:
                rot = "D"
                order = str(max_rot_order)
                if refl_num > max_rot_order:
                    refl = "h"
                elif refl_num == max_rot_order:
                    refl = "d"
                    if max_rot_order == 2:
                        min_ang = INF
                        main_idx = 0
                        for i in range(rot_num):
                            ang = intersectangle(pot_main_axis, axes[i][1])
                            if ang < min_ang:
                                min_ang = ang
                                main_idx = i
                        axes.insert(0, axes.pop(main_idx))
                transform = vecs_rot(_PRIMAX, _SECAX, axes[0][1], axes[1][1])
            elif rotorefl_num > 0 and refl_num == 0:
                rot = "S"
                order = str(2 * max_rot_order)
                transform = vec_rot(_PRIMAX, pot_main_axis)
            else:
                rot = "C"
                if max_rot_order > 1:
                    order = str(max_rot_order)
                    if refl_num == 1:
                        refl = "h"
                    elif refl_num > 1:
                        refl = "v"
                    transform = (
                        vecs_rot(_PRIMAX, _SECAX, axes[0][1], normals[0])
                        if refl == "v"
                        else vec_rot(_PRIMAX, axes[0][1])
                    )
                else:
                    if invertible:
                        refl = "i"
                    elif refl_num > 0:
                        refl = "s"
                        transform = vec_rot(_PRIMAX, normals[0])
                    else:
                        order = "1"
        return cls(rot + order + refl, transform)

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
            # Degenerate: Kh
            return cls("Kh")

        def transformation(
            axis1: Vector, axis2: Optional[Vector] = None
        ) -> Union[Identity, Rotation]:
            """
            Return the transformation from the principal axes to axes `axis1`
            and `axis2`.
            """
            if axis2 is not None:
                return vecs_rot(_PRIMAX, _SECAX, axis1, axis2)
            else:
                return vec_rot(_PRIMAX, axis1)

        # If all points are collinear, the position vectors of the points (of
        # the centered set) are linearly dependent (parallel or antiparallel).
        main_axis = _ORIGIN
        for i in range(n_points):
            pos = poses[i]
            # Select the first non-zero position vector as a potential axis to
            # test on collinearity.
            if not zero(pos, tol):
                main_axis = pos
                break
        for i in range(i + 1, n_points):
            if not parallel(main_axis, poses[i], tol):
                break
        else:
            # Linear: Coov,Dooh
            if _PRIMAX.dot(main_axis) < 0.0:
                main_axis = -main_axis
            return cls(
                "Dooh" if invertible else "Coov", transformation(main_axis)
            )

        # Define the lists of checked axes of rotation and normals of
        # reflection planes, saved as unit vectors.
        axes: List[Vector] = []
        normals: List[Vector] = []

        def new(arr: List[Vector], unitvec: Vector) -> bool:
            """
            Check whether a unit vector `unitvec` is linearly independent of
            every unit vector in a list `arr`, and add the vector to the list
            if it is.
            """
            for elem in arr:
                if unitparallel(elem, unitvec, tol):
                    return False
            arr.append(unitvec)
            return True

        # Determine the type of rotor (top) based on the principal moments of
        # inertia.
        eigvals, eigvecs = eigh(inertia(poses))
        oblate = eigvals[1] - eigvals[0] <= tol
        prolate = eigvals[2] - eigvals[1] <= tol
        if oblate and prolate:
            # Spherical: T,Td,Th,O,Oh,I,Ih
            # Define the list of found rotation axes.
            rots: List[RotationAxis] = []
            # Define the logical variable indicating the completion of the
            # search for two rotation axes.
            complete = False
            for _, idxs in points._groups:
                n_points = len(idxs)
                # Define the logical variables indicating whether all points in
                # the group are collinear or coplanar.
                collinear_group = False
                coplanar_group = False
                for i1 in range(n_points - 2):
                    pos1 = poses[idxs[i1]]
                    for i2 in range(i1 + 1, n_points - 1):
                        pos2 = poses[idxs[i2]]
                        segment = pos1 - pos2
                        # Assume the points in the group are collinear for the
                        # first pair of points.
                        if i2 == 1:
                            collinear_group = True
                        for i3 in range(i2 + 1, n_points):
                            pos3 = poses[idxs[i3]]
                            # Attempt to find a unique plane containing the
                            # triplet of points.
                            normal = cross(segment, pos1 - pos3)
                            normal_norm = norm(normal)
                            if normal_norm <= tol:
                                continue
                            # The points in the group are not collinear, since
                            # a unique plane was found.
                            collinear_group = False
                            # The normal of the plane is a potential axis of
                            # rotation.
                            axis = normal / normal_norm
                            if new(axes, axis):
                                # Calculate the distance from the origin to the
                                # plane.
                                dist = pos1.dot(axis)
                                # The number of points in the plane is the
                                # highest possible order of a potential
                                # rotation axis.  Three points are already
                                # in the plane.
                                max_order = 3
                                # All other points should be checked to avoid
                                # missing a point that was in a possible
                                # collinear triplet in previous iterations.
                                for i4 in range(n_points):
                                    if i4 != i1 and i4 != i2 and i4 != i3:
                                        pos4 = poses[idxs[i4]]
                                        # If the new point is in the plane,
                                        # increment the highest possible order.
                                        if abs(pos4.dot(axis) - dist) <= tol:
                                            max_order += 1
                                # If all points are in the plane, the points in
                                # the group are coplanar.
                                if i3 == 2 and max_order == n_points:
                                    coplanar_group = True
                                # Try all valid orders starting from the
                                # highest possible.
                                for order in range(max_order, 1, -1):
                                    # The order must be a divisor of the
                                    # highest possible order or its predecessor
                                    # (in the case that the axis contains one
                                    # point) due to symmetry.
                                    if (
                                        max_order % order == 0
                                        or (max_order - 1) % order == 0
                                    ):
                                        rot = RotationAxis(axis, order)
                                        if rot.symmetric(points, tol):
                                            rots.append(rot)
                                            # The point group can be determined
                                            # after finding only two rotation
                                            # axes.
                                            if len(rots) >= 2:
                                                complete = True
                                            break
                            if complete or coplanar_group:
                                break
                        else:
                            if not collinear_group:
                                continue
                        break
                    else:
                        continue
                    break
                # If the group is collinear or coplanar, go to the next group
                # if the search has not yet been completed.
                if complete:
                    break
            else:
                raise ValueError("no two rotation axes found")
            rot1 = rots[0]
            rot2 = rots[1]
            vec1 = rot1.vec
            vec2 = rot2.vec
            order1 = rot1.order
            order2 = rot2.order
            # Sort the rotation axes in descending order of their orders.
            if order1 < order2:
                vec1, vec2 = vec2, vec1
                order1, order2 = order2, order1
            key = (order1, order2)
            if key not in VARIANTS:
                raise ValueError(
                    f"invalid orders of rotation axes: {order1} and {order2}"
                )
            # Ensure that the angle between the vectors of the rotation axes is
            # not obtuse.
            if vec1.dot(vec2) < 0.0:
                vec2 = -vec2
            ang = angle(vec1, vec2)
            # Find the closest possible variants of point groups using the
            # orders of the rotation axes and the intersection angle between
            # them.
            min_diff = INF
            for pot_ang, pot_variants in VARIANTS[key].items():
                diff = abs(ang - pot_ang)
                if diff < min_diff:
                    min_diff = diff
                    variants = pot_variants
            # Tetrahedral, octahedral, and icosahedral point groups with an
            # inversion center contain horizontal reflection planes.
            suffix = "h" if invertible else ""
            # Construct an orthonormal set of vectors using the vectors of the
            # rotation axes.
            orthonorm_vecs = [
                vec1,
                normalize(orthogonalize(vec2, vec1)),
            ]
            orthonorm_vecs.append(
                normalize(cross(orthonorm_vecs[0], orthonorm_vecs[1]))
            )
            n_variants = len(variants)
            for i_variant in range(n_variants):
                vecs_obj, axes_order = variants[i_variant]
                # Construct a basis by adjusting the order and the signs of the
                # orthonormal vectors.
                basis = tuple(
                    (
                        orthonorm_vecs[i_axis]
                        if i_axis >= 0
                        else -orthonorm_vecs[-i_axis]
                    )
                    for i_axis in axes_order
                )
                # If the point group is tetrahedral without an inversion
                # center, check whether the point group contains diagonal
                # reflection planes by testing the system on symmetry with
                # respect to a potential four-fold rotoreflection axis.
                if suffix == "" and vecs_obj.symb == "T":
                    comps = vecs_obj.vecs[1][0]
                    rotorefl = RotoreflectionAxis(
                        comps[0] * basis[0]
                        + comps[1] * basis[1]
                        + comps[2] * basis[2],
                        4,
                    )
                    if rotorefl.symmetric(points, tol):
                        suffix = "d"
                # If it is not the last variant, test the system on symmetry
                # with repect to its first symmetry element.
                if i_variant < n_variants - 1:
                    generator = vecs_obj.symm_elems(
                        basis[0], basis[1], basis[2], suffix
                    )
                    symm_elem = next(generator)
                    if symm_elem.symmetric(points, tol):
                        break
            return cls(
                f"{vecs_obj.symb}{suffix}",
                transformation(basis[0], basis[1]),
            )
        elif oblate or prolate:
            # Symmetric: C(n>2),C(n>2)v,C(n>2)h,S(2n),D(n>3),Dnd,D(n>3)h
            # The non-degenerate principal axis is the main axis of the point
            # group.
            main_axis = eigvecs[:, 2] if oblate else eigvecs[:, 0]
            axes.append(main_axis)
            # Assume all points are coplanar.
            coplanar = True
            # Define the set of possible orders of the main rotation axis.
            orders = set()
            for _, idxs in points._groups:
                # Define the dictionary containing information about the planes
                # perpendicular to the main axis.  The keys are the distances
                # from the origin to the planes, and the values are the numbers
                # of points in the planes.
                dists: Dict[float, int] = {}
                n_points = len(idxs)
                for i in range(n_points):
                    pos = poses[idxs[i]]
                    dist = pos.dot(main_axis)
                    # If the distance from the origin to the plane is not zero,
                    # the points (of the centered set) are not coplanar.
                    if coplanar and dist > tol:
                        coplanar = False
                    # If the distance is already known, increase the number of
                    # points in the corresponding plane, and, if not, define
                    # a new plane with this distance.
                    for other_dist in dists:
                        if abs(dist - other_dist) <= tol:
                            dists[other_dist] += 1
                            break
                    else:
                        dists[dist] = 1
                # Add the numbers of points in each plane to the set of
                # possible orders.
                for count in dists.values():
                    orders.add(count)
            pot_orders = sorted(orders, reverse=True)
            # Try all valid orders starting from the highest possible.
            for order in range(pot_orders[0], 1, -1):
                for pot_order in pot_orders:
                    # The order must be a divisor of the highest possible
                    # order or its predecessor (in the case that the axis
                    # contains one point) for each set of coplanar points due
                    # to symmetry.
                    if pot_order % order != 0 and (pot_order - 1) % order != 0:
                        break
                else:
                    if RotationAxis(main_axis, order).symmetric(points, tol):
                        max_order = order
                        # Define the factor between the order of the main
                        # rotoreflection axis and the order of the main
                        # rotation axis.  If the factor remains 0, no
                        # rotoreflection axis was found.
                        rotorefl_factor = 0
                        # If the points are coplanar, the order of the main
                        # rotoreflection axis is the same as the order of the
                        # main rotation axis.  Two-fold rotoreflection axes are
                        # redundant.
                        if not coplanar:
                            # Test the possible orders of the main
                            # rotoreflection axis in descending order.
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
                        # The segment connecting two points is the normal of a
                        # potential reflection plane, and the distance vector
                        # from the origin to the segment is the axis of a
                        # potential two-fold rotation axis, if the distance
                        # vector splits the segment in two equal parts, i.e.
                        # the position vector of the midpoint is perpendicular
                        # to the segment.
                        if not perpendicular(segment, midpoint, tol):
                            continue
                        midpoint_norm = norm(midpoint)
                        nonzero = midpoint_norm > tol
                        # If the midpoint is in the origin, the axis of the
                        # potential two-fold rotation axis can be calculated
                        # for the coplanar set of points.
                        if nonzero or coplanar:
                            # For the zero position vector of the midpoint, the
                            # axis is the vector product of the segment and the
                            # normal of the plane containing the points (main
                            # axis).
                            axis = (
                                midpoint / midpoint_norm
                                if nonzero
                                else normalize(cross(segment, main_axis))
                            )
                            # The axes of two-fold rotation axes in symmetric
                            # tops are perpendicular to the main axis.
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
                        # The normals of vertical and diagonal reflection
                        # planes in symmetric tops are perpendicular to the
                        # main axis.
                        if (
                            (perpendicular(normal, main_axis, tol))
                            and new(normals, normal)
                            and ReflectionPlane(normal).symmetric(points, tol)
                        ):
                            # If the order of the main rotoreflection axis is
                            # double the order of the main rotation axis, the
                            # found reflection plane is diagonal to the
                            # two-fold rotation axes, which define the
                            # orientation of the point group.  Therefore, the
                            # normal is rotated to determine the axis of
                            # rotation.
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
                # If no other symmetry elements were found, the orientation of
                # the point group is determined only by the main axis.
                transform = transformation(main_axis)
                if rotorefl_factor == 2:
                    return cls(f"S{2*max_order}", transform)
                elif rotorefl_factor == 1:
                    return cls(f"C{max_order}h", transform)
                else:
                    return cls(f"C{max_order}", transform)
        else:
            # Asymmetric: C1,Cs,Ci,C2,C2v,C2h,D2,D2h
            # Define the list recording the numbers of found sets of symmetry
            # elements along each principal axis.  Each list index corresponds
            # to a two-figure binary number representing the presence of
            # symmetry elements in the corresponding set.  From the right, the
            # first figure indicates the presence of a reflection plane, and
            # the second figure indicates the presence of a two-fold rotation
            # axis.
            nums = 4 * [0]  # 0b00: #(), 0b01: #(s), 0b10: #(C2), 0b11: #(C2,s)
            axes = []
            for i in range(3):
                # Each principal axis is the axis of a potential two-fold
                # rotation axis or the normal of a potential reflection plane.
                vec = eigvecs[:, i]
                found_rot = RotationAxis(vec, 2).symmetric(points, tol)
                if invertible:
                    # Each two-fold rotation axis has a perpendicular
                    # reflection plane in a system with an inversion center.
                    found_refl = found_rot
                else:
                    found_refl = ReflectionPlane(vec).symmetric(points, tol)
                if found_rot or found_refl:
                    axes.append(vec)
                idx = 0  # 0 == 0b00
                if found_rot:
                    idx += 2  # 2 == 0b10
                if found_refl:
                    idx += 1  # 1 == 0b01
                nums[idx] += 1
                # The point group can be determined after analyzing only two
                # principal axes.
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
                if order > max_rot_order:
                    max_rot_order = order
                if order == 2:
                    rot2_num = num
            elif symm_elem_type is RotoreflectionAxis:
                order = prop[1]
                if order > max_rotorefl_order:
                    max_rotorefl_order = order
            elif symm_elem_type is ReflectionPlane:
                refl_num = num
            elif symm_elem_type is InversionCenter:
                invertible = True
        variants = list(_LOW_POINT_GROUPS)
        new_variants: Set[Tuple[int, int, int]] = set()

        def add(rot: str, order: int, refl: str = "") -> None:
            new_variants.add(
                (
                    Symb.PT_GRP_ROTS.index(rot),
                    order,
                    Symb.PT_GRP_REFLS.index(refl),
                )
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
            symb = (
                Symb.PT_GRP_ROTS[rot] + str(order) + Symb.PT_GRP_REFLS[refl]
            ).strip()
            group = cls(symb)
            pot_info = SymmetryElements()
            pot_info.include(tuple(group.symm_elems), TOL)
            variants.append((group, pot_info))
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
        f"C{Symb.INF}",
        f"C{Symb.INF}v",
        f"C{Symb.INF}h",
        f"D{Symb.INF}",
        f"D{Symb.INF}h",
        "K",
        "Kh",
    )
)
