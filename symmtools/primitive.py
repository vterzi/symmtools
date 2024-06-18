"""Classes for primitive objects in a real 3D space."""

__all__ = ["Point", "Points", "LabeledPoint", "Arrow", "StructPoint"]

from re import findall

from numpy import empty, zeros

from .const import INF, TOL, LABEL_RE, FLOAT_RE
from .vecop import (
    diff,
    zero,
    unitindep,
    norm,
    cross,
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

        def contains(array: List[Vector], vector: Vector) -> bool:
            for elem in array:
                if unitparallel(elem, vector, tol):
                    return True
            array.append(vector)
            return False

        symmelems: List[SymmetryElement] = []
        center = InversionCenter()
        if center.symmetric(self, tol):
            symmelems.append(center)
        axes: List[Vector] = []
        planes: List[Vector] = []
        for _, idxs in self._groups:
            n_points = len(idxs)
            collinear = False
            coplanar = False
            for i1 in range(n_points - 1):
                pos1 = poses[idxs[i1]]
                for i2 in range(i1 + 1, n_points):
                    pos2 = poses[idxs[i2]]
                    segment = pos1 - pos2
                    if not collinear and not coplanar:
                        if i2 == 1:
                            collinear = True
                        for i3 in range(i2 + 1, n_points):
                            pos3 = poses[idxs[i3]]
                            normal = cross(segment, pos1 - pos3)
                            normal_norm = norm(normal)
                            if normal_norm <= tol:
                                continue
                            collinear = False
                            axis = normal / normal_norm
                            if not contains(axes, axis):
                                dist = pos1.dot(axis)
                                max_order = 3
                                for i4 in range(i3 + 1, n_points):
                                    pos4 = poses[idxs[i4]]
                                    if abs(pos4.dot(axis) - dist) <= tol:
                                        max_order += 1
                                if i3 == 2 and max_order == n_points:
                                    coplanar = True
                                orders = set(range(max_order, 2, -1)).union(
                                    set(range(2 * max_order, 4, -2))
                                )
                                for order in sorted(orders, reverse=True):
                                    rotorefl = RotoreflectionAxis(axis, order)
                                    if rotorefl.symmetric(self, tol):
                                        symmelems.append(rotorefl)
                                        break
                                for order in range(max_order, 1, -1):
                                    rot = RotationAxis(axis, order)
                                    if rot.symmetric(self, tol):
                                        symmelems.append(rot)
                                        break
                    midpoint = 0.5 * (pos1 + pos2)
                    if not perpendicular(segment, midpoint, tol):
                        continue
                    normal = segment / norm(segment)
                    midpoint_norm = norm(midpoint)
                    if midpoint_norm > tol:
                        axis = midpoint / midpoint_norm
                    else:
                        axis = normal
                    if not contains(axes, axis):
                        rotorefl = RotoreflectionAxis(axis, 4)
                        if rotorefl.symmetric(self, tol):
                            symmelems.append(rotorefl)
                        rot = RotationAxis(axis, 2)
                        if rot.symmetric(self, tol):
                            symmelems.append(rot)
                    if not contains(planes, normal):
                        refl = ReflectionPlane(normal)
                        if refl.symmetric(self, tol):
                            symmelems.append(refl)
        n_points = len(poses)
        if n_points == 1:
            symmelems.append(CenterRotationAxes())
            symmelems.append(CenterReflectionPlanes())
            symmelems.append(CenterRotoreflectionAxes())
        else:
            for i in range(n_points):
                pos = poses[i]
                if not zero(pos, tol):
                    axis = pos
                    break
            collinear = True
            for i in range(i + 1, n_points):
                if not parallel(axis, poses[i], tol):
                    collinear = False
                    break
            if collinear:
                symmelems.append(InfRotationAxis(axis))
                symmelems.append(AxisReflectionPlanes(axis))
                if center in symmelems:
                    symmelems.append(InfRotoreflectionAxis(axis))
                    symmelems.append(AxisRotationAxes(axis))
            else:
                for i1 in range(n_points):
                    pos = poses[i1]
                    for i2 in range(i1 + 1, n_points):
                        product = cross(pos, poses[i2])
                        if not zero(normal, tol):
                            normal = product
                            break
                coplanar = True
                for i in range(i2 + 1, n_points):
                    if not perpendicular(normal, poses[i], tol):
                        coplanar = False
                        break
                if coplanar:
                    symmelems.append(ReflectionPlane(normal))
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
