"""Classes for symmetry elements in a real 3D space."""

__all__ = [
    "SymmElem",
    "IdentityElem",
    "InversionCenter",
    "RotationAxis",
    "InfRotationAxis",
    "ReflectionPlane",
    "RotoreflectionAxis",
    "InfRotoreflectionAxis",
]

from abc import ABC, abstractmethod

from .const import TAU
from .primitive import Elems
from .transform import (
    InvariantTransformable,
    DirectionTransformable,
    OrderedTransformable,
    InfFoldTransformable,
    Transformation,
    Identity,
    Inversion,
    Rotation,
    Reflection,
    Rotoreflection,
)
from .typehints import Sequence, List


class SymmElem(ABC):
    """Symmetry element in a real 3D space."""

    @abstractmethod
    def transformations(self) -> Sequence[Transformation]:
        """Return the transformations."""
        pass

    def symmetric(self, elems: Elems, tol: float) -> bool:
        """
        Check wether a set of elements `elems` is symmetric within a tolerance
        `tol`.
        """
        for transformation in self.transformations():
            if not elems.same(transformation(elems), tol):
                return False
        return True


class IdentityElem(InvariantTransformable, SymmElem):
    """Identity element."""

    def transformations(self) -> Sequence[Transformation]:
        return (Identity(),)


class InversionCenter(InvariantTransformable, SymmElem):
    """Inversion center in the origin."""

    def transformations(self) -> Sequence[Transformation]:
        return (Inversion(),)


class RotationAxis(OrderedTransformable, SymmElem):
    """Rotation axis containing the origin."""

    def transformations(self) -> Sequence[Transformation]:
        return tuple(
            Rotation(self._vec, i / self._order * TAU)
            for i in range(1, self._order)
        )


class InfRotationAxis(InfFoldTransformable, SymmElem):
    """Infinite-fold rotation axis containing the origin."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class ReflectionPlane(DirectionTransformable, SymmElem):
    """Reflection plane containing the origin."""

    def transformations(self) -> Sequence[Transformation]:
        return (Reflection(self._vec),)


class RotoreflectionAxis(OrderedTransformable, SymmElem):
    """Rotoreflection axis containing the origin."""

    def transformations(self) -> Sequence[Transformation]:
        transformations: List[Transformation] = []
        for i in range(1, self._order):
            angle = i / self._order * TAU
            transformations.append(
                Rotation(self._vec, angle)
                if i % 2 == 1
                else Rotoreflection(self._vec, angle)
            )
        return tuple(transformations)


class InfRotoreflectionAxis(InfFoldTransformable, SymmElem):
    """Infinite-fold rotoreflaction axis containing the origin."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()
