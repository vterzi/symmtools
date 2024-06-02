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

from numpy import sin, cos, empty, eye
from scipy.optimize import linear_sum_assignment  # type: ignore

from .const import INF, PI, TAU
from .vecop import (
    vector,
    norm,
    diff,
    unitindep,
    translate,
    invert,
    move2,
    reflect,
)
from .typehints import (
    TypeVar,
    Any,
    Sequence,
    Int,
    Float,
    Vector,
    Matrix,
    RealVector,
)

_Transformable = TypeVar("_Transformable", bound="Transformable")


class Transformable(ABC):
    """Transformable object."""

    def args(self) -> str:
        """Return the argument values used to create the instance."""
        return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.args()})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def diff(self, obj: Any) -> float:
        """Return the difference between the instance and an object `obj`."""
        return 0.0 if type(self) is type(obj) else INF

    def same(self, obj: Any, tol: float) -> bool:
        """
        Check wether the instance is identical to an object `obj` within a
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

    def negate(self: _Transformable) -> _Transformable:
        """Return the instance resulting from the application of a negation."""
        return self.copy()

    def symmetry(self, obj: Any, tol: float) -> int:
        """
        Return the symmetry of the instance in relation to an object `obj`
        within a tolerance `tol` (`1` for symmetric, `-1` for anti-symmetric,
        and `0` for asymmetric) if the instance is not invariant to negation,
        and otherwise check wether the instance is identical to an object `obj`
        within a tolerance `tol`.
        """
        antiself = self.negate()
        if self == antiself:
            return self.same(obj, tol)
        elif self.same(obj, tol):
            return 1
        elif antiself.same(obj, tol):
            return -1
        else:
            return 0

    @abstractmethod
    def translate(
        self: _Transformable, translation: "Translation"
    ) -> _Transformable:
        """
        Return the instance resulting from the application of a translation
        `translation`.
        """
        pass

    @abstractmethod
    def invert(self: _Transformable) -> _Transformable:
        """
        Return the instance resulting from the application of the inversion.
        """
        pass

    @abstractmethod
    def rotate(self: _Transformable, rotation: "Rotation") -> _Transformable:
        """
        Return the instance resulting from the application of a rotation
        `rotation`.
        """
        pass

    @abstractmethod
    def reflect(
        self: _Transformable, reflection: "Reflection"
    ) -> _Transformable:
        """
        Return the instance resulting from the application of a reflection
        `reflection`.
        """
        pass

    @abstractmethod
    def rotoreflect(
        self: _Transformable, rotoreflection: "Rotoreflection"
    ) -> _Transformable:
        """
        Return the instance resulting from the application of a rotoreflection
        `rotoreflection`.
        """
        pass


_Transformables = TypeVar("_Transformables", bound="Transformables")


class Transformables(Transformable):
    """Set of transformables."""

    _elems: Sequence[Transformable] = ()

    def __init__(self, elems: Sequence[Transformable]) -> None:
        """Initialize the instance with a set of elements `elems`."""
        self._elems = tuple(elems)

    @property
    def elems(self) -> Sequence[Transformable]:
        """Return the set of elements."""
        return self._elems

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

    def __getitem__(self, item: int) -> Transformable:
        return self._elems[item]

    def __len__(self) -> int:
        return len(self._elems)

    def sort(self, obj: "Transformables") -> float:
        """
        Sort the elements of the instance minimizing the difference to another
        instance `obj` and return the resulting difference.
        """
        n = len(self._elems)
        if n != len(obj.elems):
            raise ValueError("different number of elements")
        diffs = empty((n, n))
        for i1 in range(n):
            elem = self._elems[i1]
            for i2 in range(n):
                diffs[i1, i2] = elem.diff(obj.elems[i2])
        try:
            order = linear_sum_assignment(diffs)[1]
        except ValueError:
            raise ValueError("different number of elements of the same type")
        idxs = n * [n]
        for i in range(n):
            idxs[order[i]] = i
        self._elems = tuple(self._elems[idxs[i]] for i in range(n))
        diff = 0.0
        for i in range(n):
            diff = max(diff, diffs[i, order[i]])
        return diff

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            try:
                res = max(res, obj.sort(self))
            except ValueError:
                res = INF
        return res

    def nondegen(self, tol: float) -> bool:
        """
        Check wether no two elements are the same within a tolerance `tol`.
        """
        n = len(self._elems)
        for i1 in range(n - 1):
            elem = self._elems[i1]
            for i2 in range(i1 + 1, n):
                if elem.same(self._elems[i2], tol):
                    return False
        return True

    def negate(self: _Transformables) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.negate() for elem in self._elems)
        return res

    def translate(
        self: _Transformables, translation: "Translation"
    ) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.translate(translation) for elem in self._elems)
        return res

    def invert(self: _Transformables) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.invert() for elem in self._elems)
        return res

    def rotate(self: _Transformables, rotation: "Rotation") -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.rotate(rotation) for elem in self._elems)
        return res

    def reflect(
        self: _Transformables, reflection: "Reflection"
    ) -> _Transformables:
        res = self.copy()
        res._elems = tuple(elem.reflect(reflection) for elem in self._elems)
        return res

    def rotoreflect(
        self: _Transformables, rotoreflection: "Rotoreflection"
    ) -> _Transformables:
        res = self.copy()
        res._elems = tuple(
            elem.rotoreflect(rotoreflection) for elem in self._elems
        )
        return res


_InvariantTransformable = TypeVar(
    "_InvariantTransformable", bound="InvariantTransformable"
)


class InvariantTransformable(Transformable):
    """Transformable object that is invariant to any transformation."""

    def translate(
        self: _InvariantTransformable, translation: "Translation"
    ) -> _InvariantTransformable:
        return self.copy()

    def invert(self: _InvariantTransformable) -> _InvariantTransformable:
        return self.copy()

    def rotate(
        self: _InvariantTransformable, rotation: "Rotation"
    ) -> _InvariantTransformable:
        return self.copy()

    def reflect(
        self: _InvariantTransformable, reflection: "Reflection"
    ) -> _InvariantTransformable:
        return self.copy()

    def rotoreflect(
        self: _InvariantTransformable, rotoreflection: "Rotoreflection"
    ) -> _InvariantTransformable:
        return self.copy()


_VectorTransformable = TypeVar(
    "_VectorTransformable", bound="VectorTransformable"
)


class VectorTransformable(Transformable):
    """Transformable object represented by a real 3D vector."""

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a 3D vector `vec`."""
        self._vec = vector(vec)
        if self._vec.shape != (3,):
            raise ValueError("invalid vector shape")

    @property
    def vec(self) -> Vector:
        """Return the vector representing the instance."""
        return self._vec

    def args(self) -> str:
        return str(self._vec.tolist()).replace(" ", "")

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, diff(self._vec, obj.vec))
        return res

    def translate(
        self: _VectorTransformable, translation: "Translation"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = translate(self._vec, translation.vec)
        return res

    def invert(self: _VectorTransformable) -> _VectorTransformable:
        res = self.copy()
        res._vec = invert(self._vec)
        return res

    def rotate(
        self: _VectorTransformable, rotation: "Rotation"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = move2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(
        self: _VectorTransformable, reflection: "Reflection"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = reflect(self._vec, reflection.vec)
        return res

    def rotoreflect(
        self: _VectorTransformable, rotoreflection: "Rotoreflection"
    ) -> _VectorTransformable:
        res = self.copy()
        res._vec = reflect(
            move2(
                self._vec,
                rotoreflection.vec,
                rotoreflection.cos,
                rotoreflection.sin,
            ),
            rotoreflection.vec,
        )
        return res


_DirectionTransformable = TypeVar(
    "_DirectionTransformable", bound="DirectionTransformable"
)


class DirectionTransformable(VectorTransformable):
    """Transformable object represented by a real 3D direction vector."""

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a 3D non-zero vector `vec`."""
        super().__init__(vec)
        vec_norm = norm(self._vec)
        if vec_norm == 0.0:
            raise ValueError("zero vector")
        self._vec /= vec_norm

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            res = max(res, unitindep(self._vec, obj.vec))
        return res

    def translate(
        self: _DirectionTransformable, translation: "Translation"
    ) -> _DirectionTransformable:
        return self.copy()


class OrderedTransformable(DirectionTransformable):
    """
    Transformable object represented by a real 3D direction vector and an
    order.
    """

    def __init__(self, vec: RealVector, order: Int) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and a positive
        order `order`.
        """
        super().__init__(vec)
        if order < 1:
            raise ValueError("negative order")
        self._order = order

    @property
    def order(self) -> Int:
        """Return the order."""
        return self._order

    def args(self) -> str:
        return f"{super().args()},{self._order}"

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF and self._order != obj.order:
            res = INF
        return res


class InfFoldTransformable(DirectionTransformable):
    """
    Transformable object represented by a real 3D direction vector and an
    infinite order.
    """

    @property
    def order(self) -> Float:
        """Return the order."""
        return INF


_Any = TypeVar("_Any", bound=Any)
_Transformation = TypeVar("_Transformation", bound="Transformation")


class Transformation(ABC):
    """Transformation."""

    @abstractmethod
    def diff(self, obj: Any) -> float:
        pass

    @abstractmethod
    def translate(self: _Any, translation: "Translation") -> _Any:
        pass

    @abstractmethod
    def invert(self: _Any) -> _Any:
        pass

    @abstractmethod
    def rotate(self: _Any, rotation: "Rotation") -> _Any:
        pass

    @abstractmethod
    def reflect(self: _Any, reflection: "Reflection") -> _Any:
        pass

    @abstractmethod
    def rotoreflect(self: _Any, rotoreflection: "Rotoreflection") -> _Any:
        pass

    @abstractmethod
    def __call__(self, obj: _Any) -> _Any:
        """Apply the transformation."""
        pass

    @abstractmethod
    def mat(self) -> Matrix:
        """Return the transformation matrix."""
        pass


class Identity(InvariantTransformable, Transformation):
    """Identity in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.copy()

    def mat(self) -> Matrix:
        return eye(3)


class Translation(VectorTransformable, Transformation):
    """Translation in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.translate(self)

    def mat(self) -> Matrix:
        res = eye(4)
        res[:3, 3] = self._vec
        return res


class Inversion(InvariantTransformable, Transformation):
    """Inversion (point reflection) through the origin in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.invert()

    def mat(self) -> Matrix:
        return -eye(3)


class Rotation(DirectionTransformable, Transformation):
    """Rotation around an axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, angle: Float) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and a non-zero
        angle `angle`.
        """
        super().__init__(vec)
        angle %= TAU
        if angle == 0.0:
            raise ValueError("zero angle")
        self._angle = angle
        self._cos = cos(angle)
        self._sin = sin(angle)

    @property
    def angle(self) -> Float:
        """Return the rotation angle."""
        return self._angle

    @property
    def cos(self) -> Float:
        """Return the cosine of the rotation angle."""
        return self._cos

    @property
    def sin(self) -> Float:
        """Return the sine of the rotation angle."""
        return self._sin

    def args(self) -> str:
        return f"{super().args()},{self._angle}"

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.rotate(self)

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            diff1 = diff(self._vec, obj.vec)
            diff2 = diff(self._vec, -obj.vec)
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

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = move2(res[i], self._vec, self._cos, self._sin)
        return res.T


class Reflection(DirectionTransformable, Transformation):
    """Reflection through a plane containing the origin in a real 3D space."""

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.reflect(self)

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = reflect(res[i], self._vec)
        return res.T


class Rotoreflection(Rotation):
    """
    Rotoreflection around an axis containing the origin and through the
    perpendicular plane containing the origin in a real 3D space.
    """

    def __init__(self, vec: RealVector, angle: Float) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and a non-zero
        angle `angle` that is not equal to a half-turn.
        """
        super().__init__(vec, angle)
        if self._angle == PI:
            raise ValueError("half-turn angle")

    def __call__(self, obj: _Transformable) -> _Transformable:
        return obj.rotoreflect(self)

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = reflect(
                move2(res[i], self._vec, self._cos, self._sin), self._vec
            )
        return res.T
