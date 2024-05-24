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
from .typehints import Sequence, List, Int, RealVector


class SymmElem(ABC):
    """Symmetry element."""

    @abstractmethod
    def transformations(self) -> Sequence[Transformation]:
        """Return the transformations."""
        pass

    @abstractmethod
    def symb(self) -> str:
        """Return the symbol."""
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
    """Identity element in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Identity(),)

    def symb(self) -> str:
        return "E"


class InversionCenter(InvariantTransformable, SymmElem):
    """Inversion center in the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Inversion(),)

    def symb(self) -> str:
        return "i"


class RotationAxis(OrderedTransformable, SymmElem):
    """Rotation axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, order: Int) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and a positive
        order `order` greater than 1.
        """
        super().__init__(vec, order)
        if order == 1:
            raise ValueError(
                "a 1-fold rotation axis is identical to an identity element"
            )

    def transformations(self) -> Sequence[Transformation]:
        return tuple(
            Rotation(self._vec, i / self._order * TAU)
            for i in range(1, self._order)
        )

    def symb(self) -> str:
        return f"C{self._order}"


class InfRotationAxis(InfFoldTransformable, SymmElem):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return "Coo"  # "C\u221e"


class ReflectionPlane(DirectionTransformable, SymmElem):
    """Reflection plane containing the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Reflection(self._vec),)

    def symb(self) -> str:
        return "s"  # "\u03c3"


class RotoreflectionAxis(OrderedTransformable, SymmElem):
    """Rotoreflection axis containing the origin in a real 3D space."""

    def __init__(self, vec: RealVector, order: Int) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and a positive
        order `order` greater than 2.
        """
        super().__init__(vec, order)
        if order == 1:
            raise ValueError(
                "a 1-fold rotoreflection axis is identical to a reflection"
                + " plane"
            )
        elif order == 2:
            raise ValueError(
                "a 2-fold rotoreflection axis is identical to an inversion"
                + " center"
            )

    def transformations(self) -> Sequence[Transformation]:
        transformations: List[Transformation] = []
        for i in range(1, self._order * (1 if self._order % 2 == 0 else 2)):
            if i != self._order:
                angle = (i % self._order) / self._order * TAU
                if i % 2 == 0:
                    transformations.append(Rotation(self._vec, angle))
                else:
                    transformations.append(Rotoreflection(self._vec, angle))
            else:
                transformations.append(Reflection(self._vec))
        return tuple(transformations)

    def symb(self) -> str:
        return f"S{self._order}"


class InfRotoreflectionAxis(InfFoldTransformable, SymmElem):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return "Soo"  # "S\u221e"
