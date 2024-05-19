"""Classes for transformations in a real 3D space."""

__all__ = [
    "Transformable",
    "InvariantTransformable",
    "VecTransformable",
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

from numpy import sin, cos, eye
from numpy.linalg import norm

from .const import INF, TAU
from .vecop import (
    vector,
    diff,
    indepunit,
    translate,
    invert,
    move2,
    reflect,
)
from .typehints import Any, Int, Float, Vector, Matrix, RealVector


class Transformable(ABC):
    """Transformable object."""

    def args(self) -> str:
        """Return the argument values used to create the instance."""
        return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.args()})"

    def __repr__(self) -> str:
        return self.__str__()

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

    def copy(self) -> "Transformable":
        """Return a copy of the instance."""
        return copy(self)

    @abstractmethod
    def translate(self, translation: "Translation") -> "Transformable":
        """
        Return the instance resulting from the application of a translation
        `translation`.
        """
        pass

    @abstractmethod
    def invert(self) -> "Transformable":
        """
        Return the instance resulting from the application of the inversion.
        """
        pass

    @abstractmethod
    def rotate(self, rotation: "Rotation") -> "Transformable":
        """
        Return the instance resulting from the application of a rotation
        `rotation`.
        """
        pass

    @abstractmethod
    def reflect(self, reflection: "Reflection") -> "Transformable":
        """
        Return the instance resulting from the application of a reflection
        `reflection`.
        """
        pass

    @abstractmethod
    def rotoreflect(self, rotoreflection: "Rotoreflection") -> "Transformable":
        """
        Return the instance resulting from the application of a rotoreflection
        `rotoreflection`.
        """
        pass


class InvariantTransformable(Transformable):
    """Transformable object that is invariant to any transformation."""

    def translate(
        self, translation: "Translation"
    ) -> "InvariantTransformable":
        return copy(self)

    def invert(self) -> "InvariantTransformable":
        return copy(self)

    def rotate(self, rotation: "Rotation") -> "InvariantTransformable":
        return copy(self)

    def reflect(self, reflection: "Reflection") -> "InvariantTransformable":
        return copy(self)

    def rotoreflect(
        self, rotoreflection: "Rotoreflection"
    ) -> "InvariantTransformable":
        return copy(self)


class VecTransformable(Transformable):
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

    def __getitem__(self, item: Int) -> Float:
        return self._vec[item]

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, diff(self._vec, obj.vec))
        return res

    def translate(self, translation: "Translation") -> "VecTransformable":
        res = copy(self)
        res._vec = translate(self._vec, translation.vec)
        return res

    def invert(self) -> "VecTransformable":
        res = copy(self)
        res._vec = invert(self._vec)
        return res

    def rotate(self, rotation: "Rotation") -> "VecTransformable":
        res = copy(self)
        res._vec = move2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(self, reflection: "Reflection") -> "VecTransformable":
        res = copy(self)
        res._vec = reflect(self._vec, reflection.vec)
        return res

    def rotoreflect(
        self, rotoreflection: "Rotoreflection"
    ) -> "VecTransformable":
        res = copy(self)
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


class DirectionTransformable(VecTransformable):
    """Transformable object represented by a real 3D direction vector."""

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a non-zero 3D vector `vec`."""
        super().__init__(vec)
        vec_norm = norm(self._vec)
        if vec_norm == 0:
            raise ValueError("zero vector")
        self._vec /= vec_norm

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            res = max(res, indepunit(self._vec, obj.vec))
        return res

    def translate(
        self, translation: "Translation"
    ) -> "DirectionTransformable":
        return copy(self)


class OrderedTransformable(DirectionTransformable):
    """Transformable object represented by a direction vector and an order."""

    def __init__(self, vec: RealVector, order: Int) -> None:
        """
        Initialize the instance with a non-zero 3D vector `vec` and a positive
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
        return f"{super().args},{self._order}"

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF and self._order != obj.order:
            res = INF
        return res


class InfFoldTransformable(DirectionTransformable):
    """
    Transformable object represented by a direction vector and an infinite
    order.
    """

    @property
    def order(self) -> Float:
        """Return the order."""
        return INF


class Transformation(ABC):
    """Transformation in a real 3D space."""

    @abstractmethod
    def __call__(self, obj: "Transformable") -> "Transformable":
        """Apply the transformation."""
        pass

    @abstractmethod
    def mat(self) -> Matrix:
        """Return the transformation matrix."""
        pass


class Identity(InvariantTransformable, Transformation):
    """Identity in a real 3D space."""

    def __call__(self, obj: "Transformable") -> "Transformable":
        return obj.copy()

    def mat(self) -> Matrix:
        return eye(3)


class Translation(VecTransformable, Transformation):
    """Translation in a real 3D space."""

    def __call__(self, obj: "Transformable") -> "Transformable":
        return obj.translate(self)

    def mat(self) -> Matrix:
        res = eye(4)
        res[:3, 3] = self._vec
        return res


class Inversion(InvariantTransformable, Transformation):
    """Inversion (point reflection) through the origin in a real 3D space."""

    def __call__(self, obj: "Transformable") -> "Transformable":
        return obj.invert()

    def mat(self) -> Matrix:
        return -eye(3)


class Rotation(DirectionTransformable, Transformation):
    """Rotation around an axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, angle: Float) -> None:
        """
        Initialize the instance with a non-zero 3D vector `vec` and a non-zero
        angle `angle`.
        """
        super().__init__(vec)
        angle %= TAU
        if angle == 0:
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

    def __call__(self, obj: "Transformable") -> "Transformable":
        return obj.rotate(self)

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            diff1 = diff(self._vec, obj.vec)
            diff2 = diff(self._vec, -obj.vec)
            if diff1 < diff2:
                mindiff = diff1
                explementary = False
            else:
                mindiff = diff2
                explementary = True
            angle = TAU - obj.angle if explementary else obj.angle
            res = max(res, mindiff, abs(self._angle - angle))
        return res

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = move2(res[i], self._vec, self._cos, self._sin)
        return res.T


class Reflection(DirectionTransformable, Transformation):
    """Reflection through a plane containing the origin in a real 3D space."""

    def __call__(self, obj: "Transformable") -> "Transformable":
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

    def __call__(self, obj: "Transformable") -> "Transformable":
        return obj.rotoreflect(self)

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = reflect(
                move2(res[i], self._vec, self._cos, self._sin), self._vec
            )
        return res.T
