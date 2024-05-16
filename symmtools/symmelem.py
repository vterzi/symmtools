"""Classes for symmetry elements in a real 3D space."""

__all__ = [
    "SymmElem",
    "InversionCenter",
    "RotationAxis",
    "InfRotationAxis",
    "ReflectionPlane",
    "RotoreflectionAxis",
    "InfRotoreflectionAxis",
]

from abc import ABC
from copy import copy

from numpy.linalg import norm

from .const import INF, TAU
from .vecop import vector, canon, diff, invert, move2, reflect
from .primitive import Elems
from .transform import (
    Transformable,
    Transformation,
    Translation,
    Inversion,
    Rotation,
    Reflection,
    Rotoreflection,
)
from .typehints import Any, Sequence, List, Int, Float, Vector, RealVector


class SymmElem(Transformable, ABC):
    """Symmetry element in a real 3D space."""

    _transformations: Sequence[Transformation] = ()

    @property
    def transformations(self) -> Sequence[Transformation]:
        """Return the transformations."""
        return self._transformations

    def _transform(self, transformation: Transformation) -> None:
        """
        Apply a transformation `transformation` to the transformations of the
        instance.
        """
        self._transformations = tuple(
            transformation(elem) for elem in self._transformations
        )

    def symmetric(self, elems: Elems, tol: float) -> bool:
        """
        Check wether a set of elements `elems` is symmetric within a tolerance
        `tol`.
        """
        for transformation in self._transformations:
            if not elems.same(transformation(elems), tol):
                return False
        return True


class PointSymmElem(SymmElem, ABC):
    """Symmetry element in a real 3D space represented by a point."""

    def translate(self, translation: Translation) -> "PointSymmElem":
        return copy(self)

    def invert(self) -> "PointSymmElem":
        return copy(self)

    def rotate(self, rotation: Rotation) -> "PointSymmElem":
        return copy(self)

    def reflect(self, reflection: Reflection) -> "PointSymmElem":
        return copy(self)

    def rotoreflect(self, rotoreflection: Rotoreflection) -> "PointSymmElem":
        return copy(self)


class VecSymmElem(SymmElem, ABC):
    """Symmetry element in a real 3D space represented by a vector."""

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a 3D vector `vec`."""
        self._vec = vector(vec)
        if self._vec.shape != (3,):
            raise ValueError("invalid vector shape")
        self._vec = canon(self._vec)

    @property
    def vec(self) -> Vector:
        """Return the vector representing the instance."""
        return self._vec

    def __getitem__(self, item: Int) -> Float:
        return self._vec[item]

    def args(self) -> str:
        return str(list(self._vec)).replace(" ", "")

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, diff(self._vec, obj.vec))
        return res

    def invert(self) -> "VecSymmElem":
        res = copy(self)
        res._transform(Inversion())
        res._vec = invert(self._vec)
        return res

    def rotate(self, rotation: Rotation) -> "VecSymmElem":
        res = copy(self)
        res._transform(rotation)
        res._vec = move2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(self, reflection: Reflection) -> "VecSymmElem":
        res = copy(self)
        res._transform(reflection)
        res._vec = reflect(self._vec, reflection.vec)
        return res

    def rotoreflect(self, rotoreflection: Rotoreflection) -> "VecSymmElem":
        res = copy(self)
        res._transform(rotoreflection)
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


class DirectionSymmElem(VecSymmElem, ABC):
    """
    Symmetry element in a real 3D space represented by a direction vector.
    """

    def __init__(self, vec: RealVector) -> None:
        """Initialize the instance with a non-zero 3D vector `vec`."""
        super().__init__(vec)
        vec_norm = norm(self._vec)
        if vec_norm == 0:
            raise ValueError("zero vector")
        self._vec /= vec_norm

    def translate(self, translation: Translation) -> "DirectionSymmElem":
        return copy(self)


class OrderedSymmElem(DirectionSymmElem, ABC):
    """
    Symmetry element in a real 3D space represented by a direction vector and
    an order.
    """

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


class InfFoldSymmElem(DirectionSymmElem, ABC):
    """
    Symmetry element in a real 3D space represented by a direction vector and
    an infinite order.
    """

    def __init__(self, vec: RealVector) -> None:
        super().__init__(vec)
        self._order = INF

    @property
    def order(self) -> Float:
        """Return the order."""
        return self._order


class InversionCenter(PointSymmElem):
    """Inversion center in the origin in a real 3D space."""

    _transformations = (Inversion(),)


class RotationAxis(OrderedSymmElem):
    """Rotation axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, order: Int) -> None:
        """
        Initialize the instance with a non-zero 3D vector `vec` and a positive
        order `order`.
        """
        super().__init__(vec, order)
        self._transformations = tuple(
            Rotation(self._vec, i / self._order * TAU)
            for i in range(1, self._order)
        )


class InfRotationAxis(InfFoldSymmElem):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    def symmetric(self, elems: Elems, tol: float) -> bool:
        raise NotImplementedError()


class ReflectionPlane(DirectionSymmElem):
    """Reflection plane containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector) -> None:
        super().__init__(vec)
        self._transformations = (Reflection(self._vec),)


class RotoreflectionAxis(OrderedSymmElem):
    """Rotoreflection axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, order: Int) -> None:
        super().__init__(vec, order)
        transformations: List[Transformation] = []
        for i in range(1, self._order):
            angle = i / self._order * TAU
            transformations.append(
                Rotation(self._vec, angle)
                if i % 2 == 1
                else Rotoreflection(self._vec, angle)
            )
        self._transformations = tuple(transformations)


class InfRotoreflectionAxis(InfFoldSymmElem):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    def symmetric(self, elems: Elems, tol: float) -> bool:
        raise NotImplementedError()
