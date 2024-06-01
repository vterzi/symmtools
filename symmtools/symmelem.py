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


class InversionCenter(InvariantTransformable, SymmetryElement):
    """Inversion center in the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Inversion(),)

    def symb(self) -> str:
        return "i"


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


class InfRotationAxis(InfFoldTransformable, SymmetryElement):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"C{INF_SYMB}"


class ReflectionPlane(DirectionTransformable, SymmetryElement):
    """Reflection plane containing the origin in a real 3D space."""

    def transformations(self) -> Sequence[Transformation]:
        return (Reflection(self._vec),)

    def symb(self) -> str:
        return REFL_SYMB


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


class InfRotoreflectionAxis(InfFoldTransformable, SymmetryElement):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"S{INF_SYMB}"


class AxisRotationAxes(DirectionTransformable, SymmetryElement):
    """
    All two-fold rotation axes perpendicular to an infinite-fold rotation axis.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return "C2"


class CenterRotationAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotation axes containing a rotationally invariant center.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"C{INF_SYMB}"


class AxisReflectionPlanes(ReflectionPlane):
    """All reflection planes containing an infinite-fold rotation axis."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class CenterReflectionPlanes(InvariantTransformable, SymmetryElement):
    """All reflection planes containing a rotationally invariant center."""

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return REFL_SYMB


class CenterRotoreflectionAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotoreflection axes containing a rotationally invariant
    center.
    """

    def transformations(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    def symb(self) -> str:
        return f"S{INF_SYMB}"
