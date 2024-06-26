"""Classes for primitive objects in a real 3D space."""

__all__ = ["Point", "Points", "LabeledPoint", "Arrow", "StructPoint"]

from re import findall

from numpy import zeros
from numpy.linalg import eigh

from .const import INF, TOL, LABEL_RE, FLOAT_RE
from .vecop import (
    norm,
    cross,
    normalize,
    diff,
    zero,
    unitindep,
    parallel,
    unitparallel,
    perpendicular,
    inertia,
)
from .transform import (
    Transformable,
    Transformables,
    VectorTransformable,
    DirectionTransformable,
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
)
from .typehints import (
    TypeVar,
    Union,
    Any,
    Sequence,
    Tuple,
    List,
    Dict,
    Iterator,
    Bool,
    Vector,
    Matrix,
    RealVector,
    RealVectors,
)


class Point(VectorTransformable):
    """Point in a real 3D space."""

    @property
    def pos(self) -> Vector:
        """Return the position."""
        return self._vec


class LabeledPoint(Point):
    """Labeled point in a real 3D space."""

    def __init__(self, vec: RealVector, label: str) -> None:
        """
        Initialize the instance with a 3D vector `vec` and a label `label`.
        """
        super().__init__(vec)
        self._label = label

    @property
    def label(self) -> str:
        """Return the label."""
        return self._label

    @property
    def args(self):
        label = self._label.replace("'", r"\'")
        return f"{super().args},'{label}'"

    @property
    def props(self) -> Tuple:
        return super().props + (self._label,)


_Points = TypeVar("_Points", bound="Points")


class Points(Transformables):
    """Set of points."""

    _elems: Tuple[Point, ...] = ()

    def __init__(self, points: Sequence[Point]) -> None:
        """Initialize the instance with a set of points `points`."""
        super().__init__(points)

    @property
    def elems(self) -> Tuple[Point, ...]:
        """Return the set of elements."""
        return self._elems

    @property
    def pos(self) -> Vector:
        """Return the centroid of the points."""
        centroid = zeros(3)
        n = 0
        for elem in self._elems:
            centroid += elem.pos
            n += 1
        if n > 0:
            centroid /= n
        return centroid

    def __getitem__(self, item: int) -> Point:
        return self._elems[item]

    def __add__(self, other: "Points") -> "Points":
        """Return the union of the instance with a set of points `other`."""
        return type(self)(self._elems + other.elems)

    def center(self: _Points) -> _Points:
        """Center the points at the origin."""
        return self.translate(Translation(-self.pos))

    @property
    def inertia(self) -> Matrix:
        """Return the inertia tensor of the points of unit mass."""
        centroid = self.pos
        return inertia(tuple(elem.pos - centroid for elem in self._elems))

    def symm_elems(self, tol: float = TOL) -> Iterator[SymmetryElement]:
        """
        Determine all symmetry elements of a set of points `points` within a
        tolerance `tol`.
        """
        if not self.nondegen(tol):
            raise ValueError(
                "at least two identical elements in the instance of for the"
                + " given tolerance"
            )
        points = self.center()
        poses = tuple(elem.pos for elem in points._elems)

        center = InversionCenter()
        invertible = center.symmetric(points, tol)
        if invertible:
            yield center
        n_points = len(poses)

        if n_points == 1:
            yield CenterRotationAxes()
            yield CenterReflectionPlanes()
            yield CenterRotoreflectionAxes()
            return

        for i in range(n_points):
            pos = poses[i]
            if not zero(pos, tol):
                axis = pos
                break
        for i in range(i + 1, n_points):
            if not parallel(axis, poses[i], tol):
                break
        else:
            yield InfRotationAxis(axis)
            yield AxisReflectionPlanes(axis)
            if invertible:
                yield AxisRotationAxes(axis)
                yield ReflectionPlane(axis)
                yield InfRotoreflectionAxis(axis)
            return

        axes: List[Vector] = []
        normals: List[Vector] = []

        def new(arr: List[Vector], vec: Vector) -> bool:
            for elem in arr:
                if unitparallel(elem, vec, tol):
                    return False
            arr.append(vec)
            return True

        eigvals, eigvecs = eigh(inertia(poses))
        oblate = eigvals[1] - eigvals[0] <= tol
        prolate = eigvals[2] - eigvals[1] <= tol
        coplanar = False
        cubic = False
        if oblate and prolate:
            cubic = True
        elif oblate or prolate:
            # symmetric
            # C(n>2)  Cn
            # C(n>2)v Cn
            # C(n>2)h Cn,s,Sn
            # D(n>3)  Cn
            # Dnd     Cn,S(2n)
            # D(n>3)h Cn,s,Sn
            # S(2n)   Cn,S(2n)
            axis = eigvecs[:, 2] if oblate else eigvecs[:, 0]
            axes.append(axis)
            max_order = 2
            coplanar = True
            for _, idxs in points._groups:
                dists: Dict[float, int] = {}
                n_points = len(idxs)
                for i in range(n_points):
                    pos = poses[idxs[i]]
                    dist = pos.dot(axis)
                    if coplanar and dist > tol:
                        coplanar = False
                    for ref_dist in dists:
                        if abs(dist - ref_dist) <= tol:
                            dists[ref_dist] += 1
                            break
                    else:
                        dists[dist] = 1
                for count in dists.values():
                    if max_order < count:
                        max_order = count  # TODO combine different counts
            for order in range(max_order, 1, -1):
                if max_order % order != 0 and (max_order - 1) % order != 0:
                    continue
                rot = RotationAxis(axis, order)
                if rot.symmetric(points, tol):
                    yield rot
                    if not coplanar:
                        for factor in (2, 1):
                            new_order = order * factor
                            if new_order > 2:
                                rotorefl = RotoreflectionAxis(axis, new_order)
                                if rotorefl.symmetric(points, tol):
                                    yield rotorefl
                                    break
                    elif order > 2:
                        yield RotoreflectionAxis(axis, order)
                    break
            normals.append(axis)
            refl = ReflectionPlane(axis)
            if coplanar or refl.symmetric(points, tol):
                yield refl
        else:
            # asymmetric
            # C1
            # Ci
            # Cs      s
            # C2      C2
            # C2v     C2;s;s
            # C2h     C2,s
            # D2      C2;C2;C2
            # D2h     C2,s;C2,s;C2,s

            # C2,s -> C2h,D2h
            # C2,s;C2,s -> D2h(+C2,s)
            # C2,s;- -> C2h
            # C2 -> C2,C2v,D2
            # C2;C2 -> D2(+C2)
            # C2;s -> C2v(+s)
            # C2;- -> C2
            # s -> Cs,C2v
            # s;- -> Cs
            # s;s -> C2v(+C2)
            # s;C2 -> C2v(+s)
            # - -> C1,Ci,Cs,C2,C2h
            # [-;]-;C2,s -> C2h
            # [-;]-;C2 -> C2
            # [-;]-;s -> Cs
            # -;-;- -> C1,Ci
            for i in range(3):  # TODO improve
                axis = eigvecs[:, i]
                rot = RotationAxis(axis, 2)
                if rot.symmetric(points, tol):
                    yield rot
                refl = ReflectionPlane(axis)
                if refl.symmetric(points, tol):
                    yield refl
            return

        for _, idxs in points._groups:
            n_points = len(idxs)
            collinear_part = False
            coplanar_part = False
            if cubic:
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
                                        yield rot
                                        for factor in (2, 1):
                                            new_order = order * factor
                                            if new_order > 2:
                                                rotorefl = RotoreflectionAxis(
                                                    axis, new_order
                                                )
                                                if rotorefl.symmetric(
                                                    points, tol
                                                ):
                                                    yield rotorefl
                                                    break
                                        break
                        if collinear_part or coplanar_part:
                            break
                    else:
                        continue
                    break
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
                            else normalize(cross(segment, normals[0]))
                        )
                        if (
                            cubic or perpendicular(axis, axes[0], tol)
                        ) and new(axes, axis):
                            rot = RotationAxis(axis, 2)
                            if rot.symmetric(points, tol):
                                yield rot
                                rotorefl = RotoreflectionAxis(axis, 4)
                                if rotorefl.symmetric(points, tol):
                                    yield rotorefl
                    normal = normalize(segment)
                    if (cubic or perpendicular(normal, axes[0], tol)) and new(
                        normals, normal
                    ):
                        refl = ReflectionPlane(normal)
                        if refl.symmetric(points, tol):
                            yield refl

    @classmethod
    def from_arr(cls, vecs: RealVectors) -> "Points":
        """Construct an instance from an array of 3D vectors `vecs`."""
        return cls(tuple(Point(vec) for vec in vecs))

    @classmethod
    def from_str(cls, string: str) -> "Points":
        """
        Construct an instance from a string `string`.  Each three consecutive
        floating-point numbers are parsed as a `Point` instance.  If they are
        preceded by a label satisfying the rules of variable names, a
        `LabeledPoint` instance is created instead.
        """
        points = []
        for match in findall(
            r"(?:({0})\s+)?({1})\s+({1})\s+({1})".format(LABEL_RE, FLOAT_RE),
            string,
        ):
            label = match[0]
            vec = tuple(map(float, match[1:]))
            points.append(LabeledPoint(vec, label) if label else Point(vec))
        return cls(points)

    @classmethod
    def from_symm(
        cls,
        base: Union[RealVectors, "Points"],
        symm_elems: Union[SymmetryElement, Sequence[SymmetryElement]],
    ) -> "Points":
        """
        Construct an instance by applying one or multiple symmetry elements
        `symm_elems` to an array of 3D point position vectors or a set of
        points `base`.
        """
        if isinstance(base, Points):
            points = list(base.elems)
        else:
            points = [Point(vec) for vec in base]
        if not isinstance(symm_elems, Sequence):
            symm_elems = (symm_elems,)
        for symm_elem in symm_elems:
            for i in range(len(points)):
                point = points[i]
                for transform in symm_elem.transforms:
                    points.append(transform(point))
        return cls(points)


_Arrow = TypeVar("_Arrow", bound="Arrow")


class Arrow(DirectionTransformable):
    """Arrow in a real 3D space."""

    def __init__(self, vec: RealVector, fore: Bool, back: Bool) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and two
        arguments, `fore` and `back`, describing the types of its ends: `True`
        for head and `False` for tail. The direction of the vector corresponds
        the direction from the end described by `back` to the end described by
        `fore`. Possible combinations of `fore` and `back` are:
        - '<->' (two heads) for `True` and `True`,
        - '>->' (one head and one tail) for `True` and `False`,
        - '<-<' (one tail and one head) for `False` and `True`,
        - '>-<' (two tails) for `False` and `False`.
        """
        super().__init__(vec)
        if fore and back:
            self._form = 1
        elif not fore and not back:
            self._form = -1
        else:
            self._form = 0
            if back:
                self._vec = -self._vec

    @property
    def form(self) -> int:
        """
        Return the form: `1` for two heads ('<->'), `-1` for two tails ('>-<'),
        and `0` otherwise ('>->' or '<-<').
        """
        return self._form

    @property
    def args(self) -> str:
        if self._form == 1:
            fore = True
            back = True
        elif self._form == -1:
            fore = False
            back = False
        else:
            fore = True
            back = False
        return f"{super().args},{fore},{back}"

    @property
    def props(self) -> Tuple:
        return super().props + (self._form,)

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            if self._form == 0:
                res = max(res, diff(self._vec, obj.vec))
            else:
                res = max(res, unitindep(self._vec, obj.vec))
        return res

    def negate(self: _Arrow) -> _Arrow:
        res = self.copy()
        if self._form == 0:
            res._vec = -self._vec
        else:
            res._form = -self._form
        return res


_StructPoint = TypeVar("_StructPoint", bound="StructPoint")


class StructPoint(Point):
    """Point with a coefficient and a set of arrows."""

    def __init__(
        self, vec: RealVector, coef: float = 1.0, arrows: Sequence[Arrow] = ()
    ) -> None:
        """
        Initialize the instance with a 3D position vector `vec`, a non-zero
        coefficient `coef`, and a set of arrows `arrows`.
        """
        super().__init__(vec)
        if coef == 0.0:
            raise ValueError("zero coefficient")
        self._arrows = Transformables(arrows)
        if coef < 0.0:
            self._arrows = self._arrows.negate()
        self._coef = abs(coef)

    @property
    def coef(self) -> float:
        """Return the coefficient."""
        return self._coef

    @property
    def arrows(self) -> Transformables:
        """Return the set of arrows."""
        return self._arrows

    @property
    def args(self) -> str:
        return f"{super().args},{self._coef},{self._arrows.args}"

    @property
    def props(self) -> Tuple:
        return super().props + self._arrows.props

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(
                res, abs(self._coef - obj.coef), self._arrows.diff(obj.arrows)
            )
        return res

    def negate(self: _StructPoint) -> _StructPoint:
        res = super().negate()
        res._arrows = self._arrows.negate()
        return res

    def invert(self: _StructPoint) -> _StructPoint:
        res = super().invert()
        res._arrows = self._arrows.invert()
        return res

    def rotate(self: _StructPoint, rot: Rotation) -> _StructPoint:
        res = super().rotate(rot)
        res._arrows = self._arrows.rotate(rot)
        return res

    def reflect(self: _StructPoint, refl: Reflection) -> _StructPoint:
        res = super().reflect(refl)
        res._arrows = self._arrows.reflect(refl)
        return res

    def rotoreflect(
        self: _StructPoint, rotorefl: Rotoreflection
    ) -> _StructPoint:
        res = super().rotoreflect(rotorefl)
        res._arrows = self._arrows.rotoreflect(rotorefl)
        return res
