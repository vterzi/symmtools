"""Classes for primitive objects in a real 3D space."""

__all__ = ["Point", "LabeledPoint", "Points"]

from math import sin, cos
from re import split, fullmatch
from typing import (
    TypeVar,
    Union,
    Sequence,
    Tuple,
    List,
    Dict,
    Iterator,
)

from .const import (
    INF,
    PI,
    TOL,
    LABEL_RE,
    UINT_RE,
    FLOAT_RE,
    ORIGIN,
    PRIMAX,
    SECAX,
    TERTAX,
)
from .linalg3d import (
    Vector,
    Matrix,
    neg,
    add,
    sub,
    mul,
    div,
    lincomb2,
    lincomb3,
    dot,
    cross,
    norm,
    normalize,
    orthogonalize,
    zero,
    parallel,
    unitparallel,
    perpendicular,
    angle,
    inertia,
    symmeig,
)
from .transform import (
    Transformables,
    Transformation,
    VectorTransformable,
    Translation,
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
from .ptgrpinfo import VARIANTS


class Point(VectorTransformable):
    """Point in a real 3D space."""

    @property
    def pos(self) -> Vector:
        """Position."""
        return self._vec


class LabeledPoint(Point):
    """Labeled point in a real 3D space."""

    def __init__(self, vec: Vector, label: str) -> None:
        """
        Initialize the instance with a 3D vector `vec` and a label `label`.
        """
        super().__init__(vec)
        self._label = label

    @property
    def label(self) -> str:
        """Label."""
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
        return self._elems

    @property
    def pos(self) -> Vector:
        """Centroid of points."""
        centroid_x = 0.0
        centroid_y = 0.0
        centroid_z = 0.0
        n = 0
        for elem in self._elems:
            pos = elem.pos
            centroid_x += pos[0]
            centroid_y += pos[1]
            centroid_z += pos[2]
            n += 1
        if n > 0:
            scalar = 1.0 / n
            centroid_x *= scalar
            centroid_y *= scalar
            centroid_z *= scalar
        return (centroid_x, centroid_y, centroid_z)

    def __getitem__(self, item: int) -> Point:
        return self._elems[item]

    def __add__(self, other: "Points") -> "Points":
        """Return the union of the instance with a set of points `other`."""
        return type(self)(self._elems + other.elems)

    def center(self: _Points) -> _Points:
        """Center the points at the origin."""
        return self.translate(Translation(neg(self.pos)))

    @property
    def inertia(self) -> Matrix:
        """Inertia tensor of the points of unit mass."""
        centroid = self.pos
        return inertia(tuple(sub(elem.pos, centroid) for elem in self._elems))

    def symm_elems(
        self, tol: float = TOL, fast: bool = True
    ) -> Iterator[SymmetryElement]:
        """
        Determine all symmetry elements of a set of points `points` within a
        tolerance `tol`. Use the fast algorithm if `fast` is enabled.
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
        eigvals, eigvecs = symmeig(inertia(poses))
        oblate = eigvals[1] - eigvals[0] <= tol
        prolate = eigvals[2] - eigvals[1] <= tol
        coplanar = False
        cubic = False
        if oblate and prolate:
            cubic = True
        elif oblate or prolate:
            vec = eigvecs[2] if oblate else eigvecs[0]
            axes.append(vec)
            coplanar = True
            orders = set()
            for _, idxs in points._groups:
                dists: Dict[float, int] = {}
                n_points = len(idxs)
                for i in range(n_points):
                    pos = poses[idxs[i]]
                    dist = dot(pos, vec)
                    if coplanar and dist > tol:
                        coplanar = False
                    for other_dist in dists:
                        if abs(dist - other_dist) <= tol:
                            dists[other_dist] += 1
                            break
                    else:
                        dists[dist] = 1
                for count in dists.values():
                    orders.add(count)
            pot_orders = sorted(orders, reverse=True)
            for order in range(pot_orders[0], 1, -1):
                for pot_order in pot_orders:
                    if pot_order % order != 0 and (pot_order - 1) % order != 0:
                        break
                else:
                    rot = RotationAxis(vec, order)
                    if rot.symmetric(points, tol):
                        yield rot
                        max_order = order
                        if not coplanar:
                            for factor in (2, 1):
                                new_order = order * factor
                                if new_order > 2:
                                    rotorefl = RotoreflectionAxis(
                                        vec, new_order
                                    )
                                    if rotorefl.symmetric(points, tol):
                                        yield rotorefl
                                        break
                        elif order > 2:
                            yield RotoreflectionAxis(vec, order)
                        break
            normals.append(vec)
            refl = ReflectionPlane(vec)
            if coplanar or refl.symmetric(points, tol):
                yield refl
        else:
            curr_rot = False
            curr_refl = False
            for i in range(3):
                prev_rot = curr_rot
                prev_refl = curr_refl
                curr_rot = False
                curr_refl = False
                vec = eigvecs[i]
                rot = RotationAxis(vec, 2)
                if rot.symmetric(points, tol):
                    yield rot
                    curr_rot = True
                refl = ReflectionPlane(vec)
                if (invertible and curr_rot) or refl.symmetric(points, tol):
                    yield refl
                    curr_refl = True
                if i == 1:
                    vec = eigvecs[2]
                    if prev_rot and prev_refl:
                        if curr_rot and curr_refl:
                            yield RotationAxis(vec, 2)
                            yield ReflectionPlane(vec)
                    elif prev_rot:
                        if curr_rot:
                            yield RotationAxis(vec, 2)
                        elif curr_refl:
                            yield ReflectionPlane(vec)
                    elif prev_refl:
                        if curr_rot:
                            yield ReflectionPlane(vec)
                        elif curr_refl:
                            yield RotationAxis(vec, 2)
                    elif not (curr_rot or curr_refl):
                        continue
                    break
            return

        def new(arr: List[Vector], vec: Vector) -> bool:
            for elem in arr:
                if unitparallel(elem, vec, tol):
                    return False
            arr.append(vec)
            return True

        def high_symm_elems(
            idxs: Tuple[int, ...]
        ) -> Iterator[Union[RotationAxis, RotoreflectionAxis]]:
            n_points = len(idxs)
            collinear_part = False
            coplanar_part = False
            for i1 in range(n_points - 2):
                pos1 = poses[idxs[i1]]
                for i2 in range(i1 + 1, n_points - 1):
                    pos2 = poses[idxs[i2]]
                    segment = sub(pos1, pos2)
                    if i2 == 1:
                        collinear_part = True
                    for i3 in range(i2 + 1, n_points):
                        pos3 = poses[idxs[i3]]
                        normal = cross(segment, sub(pos1, pos3))
                        normal_norm = norm(normal)
                        if normal_norm <= tol:
                            continue
                        axis = div(normal, normal_norm)
                        collinear_part = False
                        if new(axes, axis):
                            dist = dot(pos1, axis)
                            max_order = 3
                            for i4 in range(n_points):
                                if i4 == i1 or i4 == i2 or i4 == i3:
                                    continue
                                pos4 = poses[idxs[i4]]
                                if abs(dot(pos4, axis) - dist) <= tol:
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
                                    if not fast:
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

        def low_symm_elems(
            idxs: Tuple[int, ...]
        ) -> Iterator[
            Union[RotationAxis, RotoreflectionAxis, ReflectionPlane]
        ]:
            n_points = len(idxs)
            for i1 in range(n_points - 1):
                pos1 = poses[idxs[i1]]
                for i2 in range(i1 + 1, n_points):
                    pos2 = poses[idxs[i2]]
                    segment = sub(pos1, pos2)
                    midpoint = mul(add(pos1, pos2), 0.5)
                    if not perpendicular(segment, midpoint, tol):
                        continue
                    midpoint_norm = norm(midpoint)
                    nonzero = midpoint_norm > tol
                    if nonzero or coplanar:
                        if nonzero:
                            axis = div(midpoint, midpoint_norm)
                        else:
                            axis = normalize(cross(segment, normals[0]))
                        if (
                            cubic or perpendicular(axis, axes[0], tol)
                        ) and new(axes, axis):
                            rot = RotationAxis(axis, 2)
                            if rot.symmetric(points, tol):
                                yield rot
                                if not fast:
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

        generator: Iterator[SymmetryElement]
        for _, idxs in points._groups:
            if fast:
                if cubic:
                    generator = high_symm_elems(idxs)
                    rot1 = next(generator)
                    vec1 = rot1.vec
                    order1 = rot1.order
                    rot2 = next(generator)
                    vec2 = rot2.vec
                    order2 = rot2.order
                    if order1 < order2:
                        vec1, vec2 = vec2, vec1
                        order1, order2 = order2, order1
                    if dot(vec1, vec2) < 0.0:
                        vec2 = neg(vec2)
                    original_axes = [
                        vec1,
                        normalize(orthogonalize(vec2, vec1)),
                    ]
                    original_axes.append(
                        normalize(cross(original_axes[0], original_axes[1]))
                    )
                    ang = angle(vec1, vec2)
                    min_diff = INF
                    for pot_ang, pot_variants in VARIANTS[
                        (order1, order2)
                    ].items():
                        diff = abs(ang - pot_ang)
                        if diff < min_diff:
                            min_diff = diff
                            variants = pot_variants
                    suffix = "h" if invertible else ""
                    n_variants = len(variants)
                    for i_variant in range(n_variants):
                        vecs_obj, axes_order = variants[i_variant]
                        permut_axes = tuple(
                            (
                                original_axes[i_axis]
                                if i_axis >= 0
                                else neg(original_axes[-i_axis])
                            )
                            for i_axis in axes_order
                        )
                        if suffix == "" and vecs_obj.symb == "T":
                            vec = vecs_obj.vecs[1][0]
                            rotorefl = RotoreflectionAxis(
                                lincomb3(
                                    permut_axes[0],
                                    vec[0],
                                    permut_axes[1],
                                    vec[1],
                                    permut_axes[2],
                                    vec[2],
                                ),
                                4,
                            )
                            if rotorefl.symmetric(points, tol):
                                suffix = "d"
                        generator = vecs_obj.symm_elems(
                            permut_axes[0],
                            permut_axes[1],
                            permut_axes[2],
                            suffix,
                        )
                        if i_variant < n_variants - 1:
                            symm_elem = next(generator)
                            if symm_elem.symmetric(points, tol):
                                yield symm_elem
                                break
                    yield from generator
                else:
                    try:
                        symm_elem = next(low_symm_elems(idxs))
                    except StopIteration:
                        continue
                    yield symm_elem
                    axis = axes[0]
                    vec1 = normalize(orthogonalize(symm_elem.vec, axis))
                    vec2 = normalize(cross(axis, vec1))
                    ang = 0.0
                    step = PI / max_order
                    if isinstance(symm_elem, RotationAxis):
                        refl = ReflectionPlane(vec1)
                        vert = refl.symmetric(points, tol)
                        if vert:
                            yield refl
                        else:
                            half_step = 0.5 * step
                            refl = ReflectionPlane(
                                lincomb2(
                                    vec1, cos(half_step), vec2, sin(half_step)
                                )
                            )
                            diag = refl.symmetric(points, tol)
                            if diag:
                                yield refl
                        for factor in range(1, max_order):
                            ang += step
                            new_vec = lincomb2(vec1, cos(ang), vec2, sin(ang))
                            yield RotationAxis(new_vec, 2)
                            if vert:
                                yield ReflectionPlane(new_vec)
                            elif diag:
                                new_ang = ang + half_step
                                yield ReflectionPlane(
                                    lincomb2(
                                        vec1, cos(new_ang), vec2, sin(new_ang)
                                    )
                                )
                    elif isinstance(symm_elem, ReflectionPlane):
                        rot = RotationAxis(vec1, 2)
                        vert = rot.symmetric(points, tol)
                        if vert:
                            yield rot
                        else:
                            half_step = 0.5 * step
                            rot = RotationAxis(
                                lincomb2(
                                    vec1, cos(half_step), vec2, sin(half_step)
                                ),
                                2,
                            )
                            diag = rot.symmetric(points, tol)
                            if diag:
                                yield rot
                        for factor in range(1, max_order):
                            ang += step
                            new_vec = lincomb2(vec1, cos(ang), vec2, sin(ang))
                            yield ReflectionPlane(new_vec)
                            if vert:
                                yield RotationAxis(new_vec, 2)
                            elif diag:
                                new_ang = ang + half_step
                                yield RotationAxis(
                                    lincomb2(
                                        vec1, cos(new_ang), vec2, sin(new_ang)
                                    ),
                                    2,
                                )
            else:
                if cubic:
                    yield from high_symm_elems(idxs)
                yield from low_symm_elems(idxs)

    @classmethod
    def from_arr(cls, vecs: Sequence[Vector]) -> "Points":
        """Construct an instance from an array of 3D vectors `vecs`."""
        return cls(tuple(Point(vec) for vec in vecs))

    @classmethod
    def from_str(
        cls,
        string: str,
        record_delim: str = "\n",
        field_delim: str = r"\s+",
    ) -> "Points":
        """
        Construct an instance from a string `string` where records are
        separated by `record_delim`, and fields within each record are
        separated by `field_delim`.  If the fields within a record are three
        floating-point numbers, they are parsed as a `Point` instance.  If they
        are preceded by a label satisfying the rules of variable names, a
        `LabeledPoint` instance is created instead.
        """
        points = []
        for record in split(record_delim, string.strip()):
            fields = tuple(
                field.strip() for field in split(field_delim, record.strip())
            )
            n_fields = len(fields)
            if n_fields == 3:
                label = ""
                floats = fields
            elif n_fields == 4:
                label = fields[0]
                if not fullmatch(LABEL_RE, label):
                    raise ValueError(f"invalid label: {label}")
                floats = fields[1:]
            else:
                raise ValueError(
                    f"invalid number of fields in record: {record}"
                )
            for field in floats:
                if not fullmatch(FLOAT_RE, field):
                    raise ValueError(f"invalid floating-point number: {field}")
            vec = (float(floats[0]), float(floats[1]), float(floats[2]))
            points.append(LabeledPoint(vec, label) if label else Point(vec))
        return cls(points)

    @classmethod
    def from_zmat(
        cls,
        string: str,
        deg: bool = True,
        record_delim: str = "\n",
        field_delim: str = r"\s+",
    ) -> "Points":
        """
        Construct an instance from a string `string` representing a Z-matrix
        where records are separated by `record_delim`, and fields within each
        record are separated by `field_delim`.  Each record is parsed as a
        `LabeledPoint` instance.  If `deg` is enabled, angles are interpreted
        as degrees.
        """
        points: List[LabeledPoint] = []
        for i_record, record in enumerate(split(record_delim, string.strip())):
            fields = tuple(
                field.strip() for field in split(field_delim, record.strip())
            )
            n_fields = len(fields)
            if n_fields != min(2 * i_record + 1, 7):
                raise ValueError(
                    f"invalid number of fields in record: {record}"
                )
            label = fields[0]
            if not fullmatch(LABEL_RE, label):
                raise ValueError(f"invalid label: {label}")
            idxs: List[int] = []
            floats: List[float] = []
            for i_field, field in enumerate(fields[1:]):
                if i_field % 2 == 0:
                    if fullmatch(UINT_RE, field):
                        i = int(field)
                        n_points = len(points)
                        if not 1 <= i <= n_points:
                            raise ValueError(
                                f"index out of bounds [1, {n_points}]: {field}"
                            )
                        idxs.append(i - 1)
                    elif fullmatch(LABEL_RE, field):
                        candidates = []
                        for i, point in enumerate(points):
                            if point.label == field:
                                candidates.append(i)
                        n_candidates = len(candidates)
                        if n_candidates == 0:
                            raise ValueError(f"unknown label: {field}")
                        elif n_candidates > 1:
                            raise ValueError(f"ambiguous label: {field}")
                        idxs.append(candidates[0])
                    else:
                        raise ValueError(f"invalid index or label: {field}")
                else:
                    if not fullmatch(FLOAT_RE, field):
                        raise ValueError(
                            f"invalid floating-point number: {field}"
                        )
                    num = float(field)
                    if i_field > 1 and deg:
                        num *= PI / 180.0
                    floats.append(num)
            n_idxs = len(idxs)
            if len(set(idxs)) != n_idxs:
                raise ValueError(f"duplicate indices in record: {record}")
            if n_idxs > 2:
                dist = floats[0]
                angle = floats[1]
                torsion = floats[2]
                vec = points[idxs[0]].pos
                vec2 = points[idxs[1]].pos
                vec1 = sub(vec2, vec)
                vec_norm = norm(vec1)
                vec1 = div(vec1, vec_norm) if vec_norm > 0.0 else PRIMAX
                vec2 = orthogonalize(sub(points[idxs[2]].pos, vec2), vec1)
                vec_norm = norm(vec2)
                vec2 = div(vec2, vec_norm) if vec_norm > 0.0 else SECAX
                vec3 = cross(vec1, vec2)
                vec_norm = norm(vec3)
                if vec_norm == 0.0:
                    vec2 = TERTAX
                    vec3 = cross(vec1, vec2)
                    vec_norm = norm(vec3)
                vec3 = div(vec3, vec_norm)
                factor = dist * sin(angle)
                vec = add(
                    vec,
                    lincomb3(
                        vec1,
                        dist * cos(angle),
                        vec2,
                        factor * cos(torsion),
                        vec3,
                        factor * sin(torsion),
                    ),
                )
            elif n_idxs == 2:
                i = idxs[0]
                dist = floats[0]
                angle = floats[1]
                comp1 = dist * cos(angle)
                comp2 = dist * sin(angle)
                if i == 1:
                    comp1 = -comp1
                vec = add(points[i].pos, lincomb2(PRIMAX, comp1, SECAX, comp2))
            elif n_idxs == 1:
                vec = add(points[idxs[0]].pos, mul(PRIMAX, floats[0]))
            elif n_idxs == 0:
                vec = ORIGIN
            points.append(LabeledPoint(vec, label))
        return cls(points)

    @classmethod
    def from_symm(
        cls,
        base: Union[Sequence[Vector], "Points"],
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

    @classmethod
    def from_transform(
        cls,
        base: Union[Sequence[Vector], "Points"],
        transforms: Union[Transformation, Sequence[Transformation]],
        tol: float,
    ) -> "Points":
        """
        Construct an instance by applying repeatedly one or multiple
        trnasfomations `transforms` to an array of 3D point position vectors or
        a set of points `base` and generating only unique points within a
        tolerance `tol`.
        """
        if isinstance(base, Points):
            points = list(base.elems)
        else:
            points = [Point(vec) for vec in base]
        if not isinstance(transforms, Sequence):
            transforms = (transforms,)
        fi = 0
        li = len(points)
        while fi < li:
            for transform in transforms:
                for i in range(fi, li):
                    point = transform(points[i])
                    for other_point in points:
                        if point.same(other_point, tol):
                            break
                    else:
                        points.append(point)
            fi = li
            li = len(points)
        return cls(points)
