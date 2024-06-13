"""Classes for symmetry elements in a real 3D space."""

__all__ = [
    "SymmetryElement",
    "IdentityElement",
    "InversionCenter",
    "RotationAxis",
    "InfRotationAxis",
    "ReflectionPlane",
    "RotoreflectionAxis",
    "InfRotoreflectionAxis",
    "AxisRotationAxes",
    "CenterRotationAxes",
    "AxisReflectionPlanes",
    "CenterReflectionPlanes",
    "CenterRotoreflectionAxes",
]

from abc import ABC, abstractmethod

from .const import TAU, INF_SYMB, REFL_SYMB
from .transform import (
    Transformable,
    Transformables,
    InvariantTransformable,
    DirectionTransformable,
    OrderedTransformable,
    InfFoldTransformable,
    Transformation,
    Inversion,
    Rotation,
    Reflection,
    Rotoreflection,
)
from .typehints import TypeVar, Sequence, List, Int, RealVector

_Transformable = TypeVar("_Transformable", bound=Transformable)


class SymmetryElement(ABC):
    """Symmetry element."""

    @abstractmethod
    def transformations(self) -> Sequence[Transformation]:
        """Return the transformations."""
        pass

    @abstractmethod
    def symb(self) -> str:
        """Return the symbol."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name."""
        pass

    @abstractmethod
    def id(self) -> int:
        """Return the ID."""
        pass

    def symmetric(self, transformables: Transformables, tol: float) -> bool:
        """
        Check wether a set of transformables `transformables` is symmetric
        within a tolerance `tol`.
        """
        for transformation in self.transformations():
            if not transformables.same(transformation(transformables), tol):
                return False
        return True

    def __call__(
        self, transformable: _Transformable
    ) -> Sequence[_Transformable]:
        """Apply the transformations."""
        return (transformable,) + tuple(
            transformation(transformable)
            for transformation in self.transformations()
        )


class IdentityElement(InvariantTransformable, SymmetryElement):
    """Identity element in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return ()

    def symb(self) -> str:
        return "E"

    def name(self) -> str:
        return "identity element"

    def id(self) -> int:
        return 1


class InversionCenter(InvariantTransformable, SymmetryElement):
    """Inversion center in the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Inversion(),)

    def symb(self) -> str:
        return "i"

    def name(self) -> str:
        return "inversion center"

    def id(self) -> int:
        return -2


class RotationAxis(OrderedTransformable, SymmetryElement):
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

    def name(self) -> str:
        return f"{self._order}-fold rotation axis"

    def id(self) -> int:
        return self._order


class InfRotationAxis(InfFoldTransformable, SymmetryElement):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"C{INF_SYMB}"

    def name(self) -> str:
        return "infinite-fold rotation axis"

    def id(self) -> int:
        return 0


class ReflectionPlane(DirectionTransformable, SymmetryElement):
    """Reflection plane containing the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Reflection(self._vec),)

    def symb(self) -> str:
        return REFL_SYMB

    def name(self) -> str:
        return "reflection plane"

    def id(self) -> int:
        return -1


class RotoreflectionAxis(OrderedTransformable, SymmetryElement):
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
                    if 2 * i != self._order:
                        transformations.append(
                            Rotoreflection(self._vec, angle)
                        )
                    else:
                        transformations.append(Inversion())
            else:
                transformations.append(Reflection(self._vec))
        return tuple(transformations)

    def symb(self) -> str:
        return f"S{self._order}"

    def name(self) -> str:
        return f"{self._order}-fold rotoreflection axis"

    def id(self) -> int:
        return -self._order


class InfRotoreflectionAxis(InfFoldTransformable, SymmetryElement):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"S{INF_SYMB}"

    def name(self) -> str:
        return "infinite-fold rotoreflection axis"

    def id(self) -> int:
        return 0


class AxisRotationAxes(DirectionTransformable, SymmetryElement):
    """
    All two-fold rotation axes perpendicular to an infinite-fold rotation axis
    containing the origin in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return "C2"

    def name(self) -> str:
        return "set of all two-fold rotation axes perpendicular to an axis"

    def id(self) -> int:
        return 2


class CenterRotationAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotation axes containing the origin as an invariant
    center in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"C{INF_SYMB}"

    def name(self) -> str:
        return "set of all infinite-fold rotation axes containing a center"

    def id(self) -> int:
        return 0


class AxisReflectionPlanes(ReflectionPlane):
    """
    All reflection planes containing an infinite-fold rotation axis and the
    origin in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def name(self) -> str:
        return "set of all reflection planes containing an axis"


class CenterReflectionPlanes(InvariantTransformable, SymmetryElement):
    """
    All reflection planes containing the origin as an invariant center in a
    real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return REFL_SYMB

    def name(self) -> str:
        return "set of all reflection planes containing a center"

    def id(self) -> int:
        return -1


class CenterRotoreflectionAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotoreflection axes containing the origin as an invariant
    center in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"S{INF_SYMB}"

    def name(self) -> str:
        return (
            "set of all infinite-fold rotoreflection axes containing a center"
        )

    def id(self) -> int:
        return 0
