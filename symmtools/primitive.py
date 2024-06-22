"""Classes for primitive objects in a real 3D space."""

__all__ = ["Point", "Points", "LabeledPoint", "Arrow", "StructPoint"]

from re import findall

from numpy import empty, zeros

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
    Any,
    Sequence,
    Tuple,
    List,
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

    _elems: Sequence[Point] = ()

    def __init__(self, points: Sequence[Point]) -> None:
        """Initialize the instance with a set of points `points`."""
        super().__init__(points)

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

    def center(self: _Points) -> _Points:
        """Center the points at the origin."""
        return self.translate(Translation(-self.pos))

    @property
    def inertia(self) -> Matrix:
        """Return the inertia tensor of the points of unit mass."""
        xx = 0.0
        yy = 0.0
        zz = 0.0
        xy = 0.0
        zx = 0.0
        yz = 0.0
        centroid = self.pos
        for elem in self._elems:
            x, y, z = elem.pos - centroid
            xs = x * x
            ys = y * y
            zs = z * z
            xx += ys + zs
            yy += zs + xs
            zz += xs + ys
            xy -= x * y
            zx -= x * z
            yz -= y * z
        mat = empty((3, 3))
        mat[0, 0] = xx
        mat[0, 1] = xy
        mat[0, 2] = zx
        mat[1, 0] = xy
        mat[1, 1] = yy
        mat[1, 2] = yz
        mat[2, 0] = zx
        mat[2, 1] = yz
        mat[2, 2] = zz
        return mat

    def symmelems(self, tol: float = TOL) -> Sequence[SymmetryElement]:
        """
        Determine all symmetry elements of a set of points `points` within a
        tolerance `tol`.
        """
        if not self.nondegen(tol):
            raise ValueError(
                "at least two identical elements in the instance of for the"
                + " given tolerance"
            )
        centroid = self.pos
        poses: Sequence[Vector] = tuple(
            elem.pos - centroid for elem in self._elems
        )
        axes: List[Vector] = []
        normals: List[Vector] = []
        symmelems: List[SymmetryElement] = []

        def new(arr: List[Vector], vec: Vector) -> bool:
            for elem in arr:
                if unitparallel(elem, vec, tol):
                    return False
            arr.append(vec)
            return True

        def add(symmelem: SymmetryElement) -> bool:
            if symmelem.symmetric(self, tol):
                symmelems.append(symmelem)
                return True
            return False

        invertible = add(InversionCenter())
        n_points = len(poses)
        if n_points == 1:
            symmelems.append(CenterRotationAxes())
            symmelems.append(CenterReflectionPlanes())
            symmelems.append(CenterRotoreflectionAxes())
            return tuple(symmelems)
        for i in range(n_points):
            pos = poses[i]
            if not zero(pos, tol):
                axis = pos
                break
        for i in range(i + 1, n_points):
            if not parallel(axis, poses[i], tol):
                break
        else:
            symmelems.append(InfRotationAxis(axis))
            symmelems.append(AxisReflectionPlanes(axis))
            if invertible:
                symmelems.append(AxisRotationAxes(axis))
                symmelems.append(ReflectionPlane(axis))
                symmelems.append(InfRotoreflectionAxis(axis))
            return tuple(symmelems)
        for i1 in range(n_points):
            pos = poses[i1]
            for i2 in range(i1 + 1, n_points):
                product = cross(pos, poses[i2])
                if not zero(product, tol):
                    normal = product
                    break
            else:
                continue
            break
        coplanar = False
        for i in range(i2 + 1, n_points):
            if not perpendicular(normal, poses[i], tol):
                break
        else:
            coplanar = True
            normal = normalize(normal)
            normals.append(normal)
            symmelems.append(ReflectionPlane(normal))
        for _, idxs in self._groups:
            n_points = len(idxs)
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
                        if not coplanar:
                            pos3 = poses[idxs[i3]]
                            normal = cross(segment, pos1 - pos3)
                            normal_norm = norm(normal)
                            if normal_norm <= tol:
                                continue
                            axis = normal / normal_norm
                        else:
                            axis = normals[0]
                        collinear_part = False
                        if new(axes, axis):
                            if not coplanar:
                                dist = pos1.dot(axis)
                                max_order = 3
                                for i4 in range(i3 + 1, n_points):
                                    pos4 = poses[idxs[i4]]
                                    if abs(pos4.dot(axis) - dist) <= tol:
                                        max_order += 1
                            else:
                                max_order = n_points
                            if i3 == 2 and max_order == n_points:
                                coplanar_part = True
                            for order in range(max_order, 1, -1):
                                if (
                                    max_order % order != 0
                                    and (max_order - 1) % order != 0
                                ):
                                    continue
                                if add(RotationAxis(axis, order)):
                                    if not coplanar:
                                        for factor in (2, 1):
                                            new_order = order * factor
                                            if new_order <= 2 or add(
                                                RotoreflectionAxis(
                                                    axis, new_order
                                                )
                                            ):
                                                break
                                    else:
                                        symmelems.append(
                                            RotoreflectionAxis(axis, order)
                                        )
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
                    normal = normalize(segment)
                    midpoint_norm = norm(midpoint)
                    nonzero = midpoint_norm > tol
                    if nonzero or coplanar:
                        axis = (
                            midpoint / midpoint_norm
                            if nonzero
                            else normalize(cross(segment, normals[0]))
                        )
                        if new(axes, axis):
                            if add(RotationAxis(axis, 2)):
                                add(RotoreflectionAxis(axis, 4))
                    if new(normals, normal):
                        add(ReflectionPlane(normal))
        return tuple(symmelems)

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
        cls, vecs: RealVectors, symm_elems: Sequence[SymmetryElement]
    ) -> "Points":
        """
        Construct an instance by applying the symmetry elements `symm_elems`
        to the array of 3D vectors `vecs`.
        """
        points = []
        for vec in vecs:
            point = Point(vec)
            points.append(point)
            for symmelem in symm_elems:
                for transform in symmelem.transforms:
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
