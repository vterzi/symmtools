"""Classes for transformations in a real 3D space."""

__all__ = [
    "Transformable",
    "Transformables",
    "InvariantTransformable",
    "VectorTransformable",
    "DirectionTransformable",
    "OrderedTransformable",
    "InfFoldTransformable",
    "Transformation",
    "Identity",
    "Translation",
    "Inversion",
    "Rotation",
    "Reflection",
    "Rotoreflection",
]

from abc import ABC, abstractmethod
from copy import copy
from math import fmod, sin, cos
from typing import (
    TypeVar,
    Any,
    Sequence,
    Tuple,
    List,
    Dict,
    Iterator,
)

from .const import INF, PI, TAU
from .linalg3d import (
    Vector,
    Matrix,
    Quaternion,
    neg,
    add,
    div,
    matmulmat,
    norm,
    diff,
    unitindep,
    trigrotate,
    reflect,
    trigrotmat,
    reflmat,
    rotquat,
)
from .utils import linassign

_Transformable = TypeVar("_Transformable", bound="Transformable")


class Transformable(ABC):
    """Transformable object."""

    @property
    def args(self) -> str:
        """Argument values used to create the instance."""
        return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.args})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

    @property
    def props(self) -> Tuple:
        """Immutable properties."""
        return (self.__class__,)

    def similar(self, obj: Any) -> bool:
        """Check whether the instance is similar to an object `obj`."""
        return isinstance(obj, Transformable) and self.props == obj.props

    def diff(self, obj: Any) -> float:
        """Return the difference between the instance and an object `obj`."""
        return 0.0 if self.similar(obj) else INF

    def same(self, obj: Any, tol: float) -> bool:
        """
        Check whether the instance is identical to an object `obj` within a
        tolerance `tol`.
        """
        return self.diff(obj) <= tol

    def __eq__(self, obj: Any) -> bool:
        return self.same(obj, 0.0)

    def __ne__(self, obj: Any) -> bool:
        return not self.same(obj, 0.0)

    def copy(self: _Transformable) -> _Transformable:
        """Return a copy of the instance."""
        return copy(self)

    @abstractmethod
    def translate(
        self: _Transformable, transl: "Translation"
    ) -> _Transformable:
        """
        Return the instance resulting from the application of a translation
        `transl`.
        """
        pass

    @abstractmethod
    def invert(self: _Transformable) -> _Transformable:
        """
        Return the instance resulting from the application of the inversion.
        """
        pass

    @abstractmethod
    def rotate(self: _Transformable, rot: "Rotation") -> _Transformable:
        """
        Return the instance resulting from the application of a rotation `rot`.
        """
        pass

    @abstractmethod
    def reflect(self: _Transformable, refl: "Reflection") -> _Transformable:
        """
        Return the instance resulting from the application of a reflection
        `refl`.
        """
        pass

    @abstractmethod
    def rotoreflect(
        self: _Transformable, rotorefl: "Rotoreflection"
    ) -> _Transformable:
        """
        Return the instance resulting from the application of a rotoreflection
        `rotorefl`.
        """
        pass


_Transformables = TypeVar("_Transformables", bound="Transformables")


class Transformables(Transformable):
    """Set of transformables."""

    _elems: Tuple[Transformable, ...] = ()

    def __init__(self, elems: Sequence[Transformable]) -> None:
        """Initialize the instance with a set of elements `elems`."""
        self._elems = tuple(elems)
        groups: Dict[Tuple, List[int]] = {}
        for i, elem in enumerate(self._elems):
            group = elem.props
            if group not in groups:
                groups[group] = []
            groups[group].append(i)
        self._groups = tuple(
            (group, tuple(groups[group])) for group in sorted(groups)
        )

    @property
    def elems(self) -> Tuple[Transformable, ...]:
        """Set of elements."""
        return self._elems

    @property
    def groups(self) -> Tuple[Tuple[Tuple, Tuple[int, ...]], ...]:
        """Group indices of elements."""
        return self._groups

    @property
    def args(self) -> str:
        return (
            "["
            + (
                ""
                if not self._elems
                else "\n  "
                + ",\n  ".join(
                    str(elem).replace("\n", "\n  ") for elem in self._elems
                )
                + ",\n"
            )
            + "]"
        )

    @property
    def props(self) -> Tuple:
        return super().props + tuple(
            (group, len(idxs)) for group, idxs in self._groups
        )

    def __getitem__(self, item: int) -> Transformable:
        return self._elems[item]

    def __len__(self) -> int:
        return len(self._elems)

    def __iter__(self) -> Iterator[Transformable]:
        return iter(self._elems)

    def order(self, other: "Transformables") -> List[int]:
        """
        Return the order of elements in an instance "other" that matches the
        order of elements in this instance.
        """
        if not self.similar(other):
            raise ValueError("dissimilar instances")
        self_elems = self._elems
        other_elems = other.elems
        order = [0] * len(self_elems)
        for (_, idxs1), (_, idxs2) in zip(self._groups, other.groups):
            diffs = []
            for idx1 in idxs1:
                elem = self_elems[idx1]
                row = []
                for idx2 in idxs2:
                    row.append(elem.diff(other_elems[idx2]))
                diffs.append(row)
            for i1, i2 in enumerate(linassign(diffs)):
                order[idxs1[i1]] = idxs2[i2]
        return order

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            try:
                for (_, idxs1), (_, idxs2) in zip(self._groups, obj.groups):
                    diffs = []
                    for idx1 in idxs1:
                        elem = self._elems[idx1]
                        row = []
                        for idx2 in idxs2:
                            row.append(elem.diff(obj[idx2]))
                        diffs.append(row)
                    for i1, i2 in enumerate(linassign(diffs)):
                        res = max(res, diffs[i1][i2])
            except ValueError:
                res = INF
        return res

    def nondegen(self, tol: float) -> bool:
        """
        Check whether no two elements are the same within a tolerance `tol`.
        """
        for _, idxs in self._groups:
            n = len(idxs)
            for i1 in range(n - 1):
                elem = self._elems[idxs[i1]]
                for i2 in range(i1 + 1, n):
                    if elem.same(self._elems[idxs[i2]], tol):
                        return False
        return True

    def translate(
        self: _Transformables, transl: "Translation"
    ) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.translate(transl) for elem in self._elems)
        return res

    def invert(self: _Transformables) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.invert() for elem in self._elems)
        return res

    def rotate(self: _Transformables, rot: "Rotation") -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.rotate(rot) for elem in self._elems)
        return res

    def reflect(self: _Transformables, refl: "Reflection") -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.reflect(refl) for elem in self._elems)
        return res

    def rotoreflect(
        self: _Transformables, rotorefl: "Rotoreflection"
    ) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.rotoreflect(rotorefl) for elem in self._elems)
        return res


_InvariantTransformable = TypeVar(
    "_InvariantTransformable", bound="InvariantTransformable"
)


class InvariantTransformable(Transformable):
    """Transformable object that is invariant to any transformation."""

    def translate(
        self: _InvariantTransformable, transl: "Translation"
    ) -> _InvariantTransformable:
        return self.copy()

    def invert(self: _InvariantTransformable) -> _InvariantTransformable:
        return self.copy()

    def rotate(
        self: _InvariantTransformable, rot: "Rotation"
    ) -> _InvariantTransformable:
        return self.copy()

    def reflect(
        self: _InvariantTransformable, refl: "Reflection"
    ) -> _InvariantTransformable:
        return self.copy()

    def rotoreflect(
        self: _InvariantTransformable, rotorefl: "Rotoreflection"
    ) -> _InvariantTransformable:
        return self.copy()


_VectorTransformable = TypeVar(
    "_VectorTransformable", bound="VectorTransformable"
)


class VectorTransformable(Transformable):
    """Transformable object represented by a real 3D vector."""

    def __init__(self, vec: Vector) -> None:
        """Initialize the instance with a vector `vec`."""
        self._vec = vec

    @property
    def vec(self) -> Vector:
        """Vector representing the instance."""
        return self._vec

    @property
    def args(self) -> str:
        return str(self._vec).replace(" ", "")

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, diff(self._vec, obj.vec))
        return res

    def translate(
        self: _VectorTransformable, transl: "Translation"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = transl.apply(self._vec)
        return res

    def invert(self: _VectorTransformable) -> _VectorTransformable:
        res = self.copy()
        res._vec = Inversion.apply(self._vec)
        return res

    def rotate(
        self: _VectorTransformable, rot: "Rotation"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = rot.apply(self._vec)
        return res

    def reflect(
        self: _VectorTransformable, refl: "Reflection"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = refl.apply(self._vec)
        return res

    def rotoreflect(
        self: _VectorTransformable, rotorefl: "Rotoreflection"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = rotorefl.apply(self._vec)
        return res


_DirectionTransformable = TypeVar(
    "_DirectionTransformable", bound="DirectionTransformable"
)


class DirectionTransformable(VectorTransformable):
    """Transformable object represented by a real 3D direction vector."""

    def __init__(self, vec: Vector) -> None:
        """Initialize the instance with a non-zero vector `vec`."""
        vec_norm = norm(vec)
        if vec_norm == 0.0:
            raise ValueError("zero vector")
        super().__init__(div(vec, vec_norm))

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            res = max(res, unitindep(self._vec, obj.vec))
        return res

    def translate(
        self: _DirectionTransformable, transl: "Translation"
    ) -> _DirectionTransformable:
        return self.copy()


class OrderedTransformable(DirectionTransformable):
    """
    Transformable object represented by a real 3D direction vector and an
    order.
    """

    def __init__(self, vec: Vector, order: int) -> None:
        """
        Initialize the instance with a non-zero vector `vec` and a positive
        order `order`.
        """
        super().__init__(vec)
        if order < 1:
            raise ValueError("negative order")
        self._order = int(order)

    @property
    def order(self) -> int:
        """Order."""
        return self._order

    @property
    def args(self) -> str:
        return f"{super().args},{self._order}"

    @property
    def props(self) -> Tuple:
        return super().props + (self._order,)


class InfFoldTransformable(DirectionTransformable):
    """
    Transformable object represented by a real 3D direction vector and an
    infinite order.
    """

    @property
    def order(self) -> float:
        """Order."""
        return INF


class Transformation(Transformable):
    """Transformation."""

    @abstractmethod
    def __call__(self, obj: _Transformable) -> _Transformable:
        """Apply the transformation."""
        pass

    @abstractmethod
    def apply(self, vec: Vector) -> Vector:
        """Apply the transformation to a vector `vec`."""
        pass

    @property
    def mat(self) -> Matrix:
        """Transformation matrix."""
        vec1 = self.apply((1.0, 0.0, 0.0))
        vec2 = self.apply((0.0, 1.0, 0.0))
        vec3 = self.apply((0.0, 0.0, 1.0))
        return (
            (vec1[0], vec2[0], vec3[0]),
            (vec1[1], vec2[1], vec3[1]),
            (vec1[2], vec2[2], vec3[2]),
        )


class Identity(InvariantTransformable, Transformation):
    """Identity in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.copy()

    @classmethod
    def apply(cls, vec: Vector) -> Vector:
        return vec

    @property
    def mat(self) -> Matrix:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class Translation(VectorTransformable, Transformation):
    """Translation in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.translate(self)

    def apply(self, vec: Vector) -> Vector:
        return add(vec, self._vec)

    @property
    def mat(self) -> Matrix:
        raise NotImplementedError("affine transformation")


class Inversion(InvariantTransformable, Transformation):
    """Inversion (point reflection) through the origin in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.invert()

    @classmethod
    def apply(cls, vec: Vector) -> Vector:
        return neg(vec)

    @property
    def mat(self) -> Matrix:
        return ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


class Rotation(DirectionTransformable, Transformation):
    """Rotation around an axis containing the origin in a real 3D space."""

    def __init__(self, vec: Vector, angle: float) -> None:
        """
        Initialize the instance with a non-zero vector `vec` and a non-zero
        angle `angle`.
        """
        super().__init__(vec)
        angle = fmod(angle, TAU)
        if angle == 0.0:
            raise ValueError("zero angle")
        angle = float(angle)
        self._angle = angle
        self._cos = cos(angle)
        self._sin = sin(angle)

    @property
    def angle(self) -> float:
        """Rotation angle."""
        return self._angle

    @property
    def cos(self) -> float:
        """Cosine of the rotation angle."""
        return self._cos

    @property
    def sin(self) -> float:
        """Sine of the rotation angle."""
        return self._sin

    @property
    def args(self) -> str:
        return f"{super().args},{self._angle}"

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.rotate(self)

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            diff1 = diff(self._vec, obj.vec)
            diff2 = diff(self._vec, neg(obj.vec))
            if diff1 < diff2:
                vec_diff = diff1
                explementary = False
            else:
                vec_diff = diff2
                explementary = True
            angle = TAU - obj.angle if explementary else obj.angle
            angle_diff = abs(self._angle - angle)
            if angle_diff > PI:
                angle_diff = TAU - angle_diff
            res = max(res, vec_diff, angle_diff)
        return res

    def apply(self, vec: Vector) -> Vector:
        return trigrotate(vec, self._vec, self._cos, self._sin)

    @property
    def mat(self) -> Matrix:
        return trigrotmat(self._vec, self._cos, self._sin)

    @property
    def quat(self) -> Quaternion:
        """Quaternion representation."""
        return rotquat(self._vec, self._angle)


class Reflection(DirectionTransformable, Transformation):
    """Reflection through a plane containing the origin in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.reflect(self)

    def apply(self, vec: Vector) -> Vector:
        return reflect(vec, self._vec)

    @property
    def mat(self) -> Matrix:
        return reflmat(self._vec)


class Rotoreflection(Rotation):
    """
    Rotoreflection around an axis containing the origin and through the
    perpendicular plane containing the origin in a real 3D space.
    """

    def __init__(self, vec: Vector, angle: float) -> None:
        """
        Initialize the instance with a non-zero vector `vec` and a non-zero
        angle `angle` that is not equal to a half-turn.
        """
        super().__init__(vec, angle)
        if self._angle == PI:
            raise ValueError("half-turn angle")

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.rotoreflect(self)

    def apply(self, vec: Vector) -> Vector:
        return reflect(
            trigrotate(vec, self._vec, self._cos, self._sin), self._vec
        )

    @property
    def mat(self) -> Matrix:
        return matmulmat(
            reflmat(self._vec), trigrotmat(self._vec, self._cos, self._sin)
        )
