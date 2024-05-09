"""Classes for transformations in a real 3D space."""

__all__ = [
    "Identity",
    "Translation",
    "Inversion",
    "Rotation",
    "Reflection",
    "Rotoreflection",
]

from abc import ABC, abstractmethod
from copy import copy
from typing import Any

from numpy import pi, sin, cos, eye
from numpy.linalg import norm

from .vecop import vector, same, parallel, invert, move2, reflect
from .typehints import Bool, Int, Float, Vector, Matrix, RealVector


class Transformation(ABC):
    """Transformation in a real 3D space."""

    @property
    def args(self) -> str:
        """Return the arguments used to create the instance."""
        return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.args})"

    def __repr__(self) -> str:
        return self.__str__()

    def same(self, obj: "Transformation", tol: Float) -> Bool:
        """
        Check wether a transformation `obj` is identical to the instance within
        a tolerance `tol`.
        """
        return type(self) is type(obj)

    @abstractmethod
    def mat(self) -> Matrix:
        """Return the transformation matrix."""
        pass

    def copy(self) -> "Transformation":
        """Return a copy of the transformation."""
        return copy(self)

    def invert(self) -> "Transformation":
        """
        Return the transformation resulting from the application of an
        inversion.
        """
        return self.copy()

    def rotate(self, rotation: "Rotation") -> "Transformation":
        """
        Return the transformation resulting from the application of a rotation
        `rotation`.
        """
        return self.copy()

    def reflect(self, reflection: "Reflection") -> "Transformation":
        """
        Return the transformation resulting from the application of a
        reflection `reflection`.
        """
        return self.copy()

    def rotoreflect(
        self, rotoreflection: "Rotoreflection"
    ) -> "Transformation":
        """
        Return the transformation resulting from the application of a
        rotoreflection `rotoreflection`.
        """
        return self.copy()

    def transform(self, transformation: "Transformation") -> "Transformation":
        """
        Return the transformation resulting from the application of a
        transformation `transformation`.
        """
        transformation_type = type(transformation)
        if transformation_type is Identity:
            return self.copy()
        elif transformation_type is Inversion:
            return self.invert()
        elif transformation_type is Rotation:
            return self.rotate(transformation)
        elif transformation_type is Reflection:
            return self.reflect(transformation)
        elif transformation_type is Rotoreflection:
            return self.rotoreflect(transformation)
        else:
            raise TypeError(f"illegal transformation: {transformation_type}")


class VecTransformation(Transformation, ABC):
    """Transformation in a real 3D space represented by a vector."""

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a 3D vector `vec`."""
        self._vec = vector(vec)
        if self._vec.shape != (3,):
            raise ValueError("invalid vector shape")

    def __getitem__(self, item: Int) -> Float:
        return self._vec[item]

    @property
    def args(self) -> str:
        return str(list(self._vec)).replace(" ", "")

    @property
    def vec(self) -> Vector:
        """Return the vector representing the transformation."""
        return self._vec

    def invert(self) -> "VecTransformation":
        res = self.copy()
        res._vec = invert(self._vec)
        return res

    def rotate(self, rotation: "Rotation") -> "VecTransformation":
        res = self.copy()
        res._vec = move2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(self, reflection: "Reflection") -> "VecTransformation":
        res = self.copy()
        res._vec = reflect(self._vec, reflection.vec)
        return res

    def rotoreflect(
        self, rotoreflection: "Rotoreflection"
    ) -> "VecTransformation":
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


class UnitVecTransformation(VecTransformation, ABC):
    """Transformation in a real 3D space represented by a unit vector."""

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a non-zero 3D vector `vec`."""
        super().__init__(vec)
        vec_norm = norm(self._vec)
        if vec_norm == 0:
            raise ValueError("zero vector")
        self._vec /= vec_norm

    def same(self, transformation: "Transformation", tol: Float) -> Bool:
        return super().same(transformation, tol) and parallel(
            self._vec, transformation.vec, tol
        )


class Identity(Transformation):
    """Identity in a real 3D space."""

    def mat(self) -> Matrix:
        return eye(3)


class Translation(VecTransformation):
    """Translation in a real 3D space."""

    def same(self, transformation: "Transformation", tol: Float) -> Bool:
        return super().same(transformation, tol) and same(
            self._vec, transformation.vec, tol
        )


class Inversion(Transformation):
    """Inversion (point reflection) through the origin in a real 3D space."""

    def mat(self) -> Matrix:
        return -eye(3)


class Rotation(UnitVecTransformation):
    """Rotation around an axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, order: Int) -> None:
        """
        Initialize the instance with a non-zero 3D vector `vec` and an order.
        """
        super().__init__(vec)
        self._order = order
        angle = 2 * pi / order
        self._cos = cos(angle)
        self._sin = sin(angle)

    @property
    def args(self) -> str:
        return f"{super().args},{self._order}"

    @property
    def order(self) -> Int:
        """Return the order of the rotation."""
        return self._order

    @property
    def cos(self) -> Float:
        """Return the cosine of the rotation angle."""
        return self._cos

    @property
    def sin(self) -> Float:
        """Return the sine of the rotation angle."""
        return self._sin

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = move2(res[i], self._vec, self._cos, self._sin)
        return res.T


class Reflection(UnitVecTransformation):
    """Reflection through a plane containing the origin in a real 3D space."""

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

    def mat(self) -> Matrix:
        res = eye(3)
        for i in range(len(res)):
            res[i] = reflect(
                move2(res[i], self._vec, self._cos, self._sin), self._vec
            )
        return res.T
