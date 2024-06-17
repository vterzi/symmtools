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

    _symb = ""
    _name = ""
    _id = 0

    @property
    @abstractmethod
    def transforms(self) -> Sequence[Transformation]:
        """Return the transformations."""
        pass

    @property
    def symb(self) -> str:
        """Return the symbol."""
        return self._symb

    @property
    def name(self) -> str:
        """Return the name."""
        return self._name

    @property
    def id(self) -> int:
        """Return the ID."""
        return self._id

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

    _symb = "E"
    _name = "identity element"
    _id = 1

    @property
    def transforms(self) -> Sequence[Transformation]:
        return ()


class InversionCenter(InvariantTransformable, SymmetryElement):
    """Inversion center in the origin in a real 3D space."""

    _symb = "i"
    _name = "inversion center"
    _id = -2

    @property
    def transforms(self) -> Sequence[Transformation]:
        return (Inversion(),)


class RotationAxis(OrderedTransformable, SymmetryElement):
    """Rotation axis containing the origin in a real 3D space."""

    _symb = SYMB.rot
    _name = "rotation axis"

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
        order_str = str(order)
        self._symb += order_str
        self._name = order_str + "-fold " + self._name
        self._id = self._order

    @property
    def transforms(self) -> Sequence[Transformation]:
        return tuple(
            Rotation(self._vec, i / self._order * TAU)
            for i in range(1, self._order)
        )


class InfRotationAxis(InfFoldTransformable, SymmetryElement):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    _symb = SYMB.rot + SYMB.inf
    _name = "infinite-fold rotation axis"

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class ReflectionPlane(DirectionTransformable, SymmetryElement):
    """Reflection plane containing the origin in a real 3D space."""

    _symb = SYMB.refl
    _name = "reflection plane"
    _id = -1

    @property
    def transforms(self) -> Sequence[Transformation]:
        return (Reflection(self._vec),)


class RotoreflectionAxis(OrderedTransformable, SymmetryElement):
    """Rotoreflection axis containing the origin in a real 3D space."""

    _symb = SYMB.rotorefl
    _name = "rotoreflection axis"

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
        order_str = str(order)
        self._symb += order_str
        self._name = order_str + "-fold " + self._name
        self._id = -self._order

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


class InfRotoreflectionAxis(InfFoldTransformable, SymmetryElement):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    _symb = SYMB.rotorefl + SYMB.inf
    _name = "infinite-fold rotoreflection axis"

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class AxisRotationAxes(DirectionTransformable, SymmetryElement):
    """
    All two-fold rotation axes perpendicular to an infinite-fold rotation axis
    containing the origin in a real 3D space.
    """

    _symb = SYMB.inf + SYMB.rot + "2"
    _name = "set of all two-fold rotation axes perpendicular to an axis"
    _id = 2

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class CenterRotationAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotation axes containing the origin as an invariant
    center in a real 3D space.
    """

    _symb = SYMB.inf + SYMB.rot + SYMB.inf
    _name = "set of all infinite-fold rotation axes containing a center"

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class AxisReflectionPlanes(ReflectionPlane):
    """
    All reflection planes containing an infinite-fold rotation axis and the
    origin in a real 3D space.
    """

    _symb = SYMB.inf + SYMB.refl + "v"
    _name = "set of all reflection planes containing an axis"

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class CenterReflectionPlanes(InvariantTransformable, SymmetryElement):
    """
    All reflection planes containing the origin as an invariant center in a
    real 3D space.
    """

    _symb = SYMB.inf + SYMB.refl
    _name = "set of all reflection planes containing a center"
    _id = -1

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


class CenterRotoreflectionAxes(InvariantTransformable, SymmetryElement):
    """
    All infinite-fold rotoreflection axes containing the origin as an invariant
    center in a real 3D space.
    """

    _symb = SYMB.inf + SYMB.rotorefl + SYMB.inf
    _name = "set of all infinite-fold rotoreflection axes containing a center"

    @property
    def transforms(self) -> Sequence[Transformation]:
        raise NotImplementedError()


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
