"""Classes for transformations in 3D space."""

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

from .vecop import vector, normalize, same, parallel, move2, reflect, invert
from .typehints import Float, Matrix


class Transformation(ABC):
    def args(self) -> str:
        """Return arguments used to create the instance."""
        return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.args()})"

    def __repr__(self) -> str:
        return self.__str__()

    def same(self, obj: Any, tol: Float) -> bool:
        return type(self) is type(obj)

    @abstractmethod
    def mat(self) -> Matrix:
        pass

    def transform(self, transformation: "Transformation"):
        type_transform = type(transformation)
        if type_transform == Identity:
            return self.copy()
        elif type_transform == Inversion:
            return self.invert()
        elif type_transform == Rotation:
            return self.rotate(transformation)
        elif type_transform == Reflection:
            return self.reflect(transformation)
        elif type_transform == Rotoreflection:
            return self.rotoreflect(transformation)
        else:
            raise ValueError(f"illegal transformation: {type_transform}")

    def copy(self) -> "Transformation":
        return copy(self)

    def invert(self) -> "Transformation":
        return self.copy()

    def rotate(self, rotation: "Rotation") -> "Transformation":
        return self.copy()

    def reflect(self, reflection: "Reflection") -> "Transformation":
        return self.copy()

    def rotoreflect(
        self, rotoreflection: "Rotoreflection"
    ) -> "Transformation":
        return self.copy()


class VecTransform(Transformation, ABC):
    def __init__(self, vec):
        self._vec = vector(vec)

    def __getitem__(self, item):
        return self._vec[item]

    def args(self):
        return str(list(self._vec)).replace(" ", "")

    @property
    def vec(self):
        return self._vec

    def invert(self):
        res = self.copy()
        res._vec = invert(self._vec)
        return res

    def rotate(self, rotation):
        res = self.copy()
        res._vec = move2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(self, reflection):
        res = self.copy()
        res._vec = reflect(self._vec, reflection.vec)
        return res

    def rotoreflect(self, rotoreflection):
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


class UnitVecTransform(VecTransform, ABC):
    def __init__(self, vec):
        super().__init__(vec)
        self._vec = normalize(self._vec)

    def same(self, transformation, tol):
        return super().same(transformation, tol) and parallel(
            self._vec, transformation.vec, tol
        )


class Identity(Transformation):
    def mat(self):
        return eye(3)


class Translation(VecTransform):
    def same(self, transformation, tol):
        return super().same(transformation, tol) and same(
            self._vec, transformation.vec, tol
        )

    def mat(self):
        return None


class Inversion(Transformation):
    def mat(self):
        return -eye(3)


class Rotation(UnitVecTransform):
    def __init__(self, vec, order):
        super().__init__(vec)
        self._order = order
        angle = 2 * pi / order
        self._cos = cos(angle)
        self._sin = sin(angle)

    def args(self):
        return f"{super().args()},{self._order}"

    @property
    def order(self):
        return self._order

    @property
    def cos(self):
        return self._cos

    @property
    def sin(self):
        return self._sin

    def mat(self):
        res = eye(3)
        for i in range(len(res)):
            res[i] = move2(res[i], self._vec, self._cos, self._sin)
        return res.T


class Reflection(UnitVecTransform):
    def mat(self):
        res = eye(3)
        for i in range(len(res)):
            res[i] = reflect(res[i], self._vec)
        return res.T


class Rotoreflection(Rotation):
    def mat(self):
        res = eye(3)
        for i in range(len(res)):
            res[i] = reflect(
                move2(res[i], self._vec, self._cos, self._sin), self._vec
            )
        return res.T
