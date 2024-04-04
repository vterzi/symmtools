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

from numpy import pi, sin, cos, eye

from .vecop import vector, normalize, same, parallel, move2, reflect, invert

tau = 2 * pi


class Transform(ABC):
    def args(self):
        return ""

    def __str__(self):
        return f"{self.__class__.__name__}({self.args()})"

    def __repr__(self):
        return self.__str__()

    def same(self, obj, tol):
        return type(self) == type(obj)

    @abstractmethod
    def mat(self):
        pass

    def transform(self, transform):
        type_transform = type(transform)
        if type_transform == Identity:
            return self.copy()
        elif type_transform == Inversion:
            return self.invert()
        elif type_transform == Rotation:
            return self.rotate(transform)
        elif type_transform == Reflection:
            return self.reflect(transform)
        elif type_transform == Rotoreflection:
            return self.rotoreflect(transform)
        else:
            raise ValueError(f"illegal transformation: {type_transform}")

    def copy(self):
        return copy(self)

    def invert(self):
        return self.copy()

    def rotate(self, rotation):
        return self.copy()

    def reflect(self, reflection):
        return self.copy()

    def rotoreflect(self, rotoreflection):
        return self.copy()


class VecTransform(Transform, ABC):
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

    def same(self, transform, tol):
        return super().same(transform, tol) and parallel(
            self._vec, transform.vec, tol
        )


class Identity(Transform):
    def mat(self):
        return eye(3)


class Translation(VecTransform):
    def same(self, transform, tol):
        return super().same(transform, tol) and same(
            self._vec, transform.vec, tol
        )

    def mat(self):
        return None


class Inversion(Transform):
    def mat(self):
        return -eye(3)


class Rotation(UnitVecTransform):
    def __init__(self, vec, order):
        super().__init__(vec)
        self._order = order
        angle = tau / order
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
