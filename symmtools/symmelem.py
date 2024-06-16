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
    "symmelems2nums",
    "symmelems2symbs",
]

from abc import ABC, abstractmethod

from .const import TAU, SYMB
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
from .typehints import (
    TypeVar,
    Type,
    Sequence,
    Tuple,
    List,
    Dict,
    Int,
    RealVector,
)

_Transformable = TypeVar("_Transformable", bound=Transformable)


class SymmetryElement(ABC):
    """Symmetry element."""

    @property
    @abstractmethod
    def transforms(self) -> Sequence[Transformation]:
        """Return the transformations."""
        pass

    @property
    @abstractmethod
    def symb(self) -> str:
        """Return the symbol."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name."""
        pass

    @property
    @abstractmethod
    def id(self) -> int:
        """Return the ID."""
        pass

    def symmetric(self, transformables: Transformables, tol: float) -> bool:
        """
        Check wether a set of transformables `transformables` is symmetric
        within a tolerance `tol`.
        """
        for transform in self.transforms:
            if not transformables.same(transform(transformables), tol):
                return False
        return True

    def __call__(
        self, transformable: _Transformable
    ) -> Sequence[_Transformable]:
        """Apply the transformations."""
        return (transformable,) + tuple(
            transform(transformable) for transform in self.transforms
        )


class IdentityElement(InvariantTransformable, SymmetryElement):
    """Identity element in a real 3D space."""

    @property
    def transforms(self) -> Sequence[Transformation]:
        return ()

    @property
    def symb(self) -> str:
        return "E"

    @property
    def name(self) -> str:
        return "identity element"

    @property
    def id(self) -> int:
        return 1


class InversionCenter(InvariantTransformable, SymmetryElement):
    """Inversion center in the origin in a real 3D space."""

    @property
    def transforms(self) -> Sequence[Transformation]:
        return (Inversion(),)

    @property
    def symb(self) -> str:
        return "i"

    @property
    def name(self) -> str:
        return "inversion center"

    @property
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

    @property
    def transforms(self) -> Sequence[Transformation]:
        return tuple(
            Rotation(self._vec, i / self._order * TAU)
            for i in range(1, self._order)
        )

    @property
    def symb(self) -> str:
        return f"C{self._order}"

    @property
    def name(self) -> str:
        return f"{self._order}-fold rotation axis"

    @property
    def id(self) -> int:
        return self._order


class InfRotationAxis(InfFoldTransformable, SymmetryElement):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"C{SYMB.inf}"

    @property
    def name(self) -> str:
        return "infinite-fold rotation axis"

    @property
    def id(self) -> int:
        return 0


class ReflectionPlane(DirectionTransformable, SymmetryElement):
    """Reflection plane containing the origin in a real 3D space."""

    @property
    def transforms(self) -> Sequence[Transformation]:
        return (Reflection(self._vec),)

    @property
    def symb(self) -> str:
        return SYMB.refl

    @property
    def name(self) -> str:
        return "reflection plane"

    @property
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

    @property
    def transforms(self) -> Sequence[Transformation]:
        res: List[Transformation] = []
        for i in range(1, self._order * (1 if self._order % 2 == 0 else 2)):
            if i != self._order:
                angle = (i % self._order) / self._order * TAU
                if i % 2 == 0:
                    res.append(Rotation(self._vec, angle))
                else:
                    if 2 * i != self._order:
                        res.append(Rotoreflection(self._vec, angle))
                    else:
                        res.append(Inversion())
            else:
                res.append(Reflection(self._vec))
        return tuple(res)

    @property
    def symb(self) -> str:
        return f"S{self._order}"

    @property
    def name(self) -> str:
        return f"{self._order}-fold rotoreflection axis"

    @property
    def id(self) -> int:
        return -self._order


class InfRotoreflectionAxis(InfFoldTransformable, SymmetryElement):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"S{SYMB.inf}"

    @property
    def name(self) -> str:
        return "infinite-fold rotoreflection axis"

    @property
    def id(self) -> int:
        return 0


class AxisRotationAxes(DirectionTransformable, SymmetryElement):
    """
    All two-fold rotation axes perpendicular to an infinite-fold rotation axis
    containing the origin in a real 3D space.
    """

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"{SYMB.inf}C2"

    @property
    def name(self) -> str:
        return "set of all two-fold rotation axes perpendicular to an axis"

    @property
    def id(self) -> int:
        return 2


class CenterRotationAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotation axes containing the origin as an invariant
    center in a real 3D space.
    """

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"{SYMB.inf}C{SYMB.inf}"

    @property
    def name(self) -> str:
        return "set of all infinite-fold rotation axes containing a center"

    @property
    def id(self) -> int:
        return 0


class AxisReflectionPlanes(ReflectionPlane):
    """
    All reflection planes containing an infinite-fold rotation axis and the
    origin in a real 3D space.
    """

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"{SYMB.inf}{SYMB.refl}v"

    @property
    def name(self) -> str:
        return "set of all reflection planes containing an axis"


class CenterReflectionPlanes(InvariantTransformable, SymmetryElement):
    """
    All reflection planes containing the origin as an invariant center in a
    real 3D space.
    """

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"{SYMB.inf}{SYMB.refl}"

    @property
    def name(self) -> str:
        return "set of all reflection planes containing a center"

    @property
    def id(self) -> int:
        return -1


class CenterRotoreflectionAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotoreflection axes containing the origin as an invariant
    center in a real 3D space.
    """

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()

    @property
    def symb(self) -> str:
        return f"{SYMB.inf}S{SYMB.inf}"

    @property
    def name(self) -> str:
        return (
            "set of all infinite-fold rotoreflection axes containing a center"
        )

    @property
    def id(self) -> int:
        return 0


def symmelems2nums(
    symmelems: Sequence[SymmetryElement],
) -> Dict[Tuple[Type[SymmetryElement], int], int]:
    """
    Count the numbers of different types of symmetry elements `symmelems`.
    """
    nums = {}
    for symmelem in symmelems:
        key = (type(symmelem), symmelem.id)
        if key not in nums:
            nums[key] = 0
        nums[key] += 1
    return nums


def symmelems2symbs(
    symmelems: Sequence[SymmetryElement], delim: str = ","
) -> str:
    """
    Concatenate the symbols of symmetry elements `symmelems` to a sorted
    string.
    """
    nums = {}
    for symmelem in symmelems:
        key = (symmelem.id, symmelem.symb)
        if key not in nums:
            nums[key] = 0
        nums[key] += 1
    string = ""
    for key, num in sorted(nums.items(), reverse=True):
        string += (str(num) if num > 1 else "") + key[1] + delim
    return string[: -len(delim)]
