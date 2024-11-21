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
    "SymmetryElements",
    "VectorSymmetryElement",
    "VEC_SYMM_ELEMS",
    "labeled_symm_elem",
    "symm_elem_rank",
]

from abc import abstractmethod

from .const import PI, TAU, Symb, SPECIAL_ANGLES
from .utils import intersectangle, rational
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
    Union,
    TypeVar,
    Sequence,
    FrozenSet,
    Tuple,
    List,
    Dict,
    Iterator,
    Int,
    RealVector,
)

_Transformable = TypeVar("_Transformable", bound=Transformable)


class SymmetryElement(Transformable):
    """Symmetry element."""

    _symb = ""
    _name = ""
    _id = 0
    label = ""

    @property
    @abstractmethod
    def transforms(self) -> Iterator[Transformation]:
        """Transformations."""
        pass

    @property
    def symb(self) -> str:
        """Symbol."""
        return self._symb

    @property
    def name(self) -> str:
        """Name."""
        return self._name

    @property
    def id(self) -> int:
        """Identifier."""
        return self._id

    def __iter__(self) -> Iterator[Transformation]:
        return self.transforms

    def symmetric(self, transformables: Transformables, tol: float) -> bool:
        """
        Check whether a set of transformables `transformables` is symmetric
        within a tolerance `tol`.
        """
        for transform in self.transforms:
            if not transformables.same(transform(transformables), tol):
                return False
        return True

    def __call__(
        self, transformable: _Transformable
    ) -> Tuple[_Transformable, ...]:
        """Apply the transformations."""
        return (transformable,) + tuple(
            transform(transformable) for transform in self.transforms
        )


class InfSymmetryElement(SymmetryElement):
    """Symmetry element with an infinite number of transformations."""

    @property
    def transforms(self) -> Iterator[Transformation]:
        raise NotImplementedError("infinite number of transformations")


class IdentityElement(InvariantTransformable, SymmetryElement):
    """Identity element in a real 3D space."""

    _symb = Symb.IDENT
    _name = "identity element"
    _id = 1

    @property
    def transforms(self) -> Iterator[Transformation]:
        return iter(())


class InversionCenter(InvariantTransformable, SymmetryElement):
    """Inversion center in the origin in a real 3D space."""

    _symb = Symb.INV
    _name = "inversion center"
    _id = -2

    @property
    def transforms(self) -> Iterator[Transformation]:
        yield Inversion()


class RotationAxis(OrderedTransformable, SymmetryElement):
    """Rotation axis containing the origin in a real 3D space."""

    _symb = Symb.ROT
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
    def transforms(self) -> Iterator[Transformation]:
        for i in range(1, self._order):
            yield Rotation(self._vec, i / self._order * TAU)


class InfRotationAxis(InfFoldTransformable, InfSymmetryElement):
    """Infinite-fold rotation axis containing the origin in a real 3D space."""

    _symb = Symb.ROT + Symb.INF
    _name = "infinite-fold rotation axis"


class ReflectionPlane(DirectionTransformable, SymmetryElement):
    """Reflection plane containing the origin in a real 3D space."""

    _symb = Symb.REFL
    _name = "reflection plane"
    _id = -1

    @property
    def transforms(self) -> Iterator[Transformation]:
        yield Reflection(self._vec)


class RotoreflectionAxis(OrderedTransformable, SymmetryElement):
    """Rotoreflection axis containing the origin in a real 3D space."""

    _symb = Symb.ROTOREFL
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
    def transforms(self) -> Iterator[Transformation]:
        for i in range(1, self._order * (1 if self._order % 2 == 0 else 2)):
            if i != self._order:
                angle = (i % self._order) / self._order * TAU
                if i % 2 == 0:
                    yield Rotation(self._vec, angle)
                else:
                    if 2 * i != self._order:
                        yield Rotoreflection(self._vec, angle)
                    else:
                        yield Inversion()
            else:
                yield Reflection(self._vec)


class InfRotoreflectionAxis(InfFoldTransformable, InfSymmetryElement):
    """
    Infinite-fold rotoreflaction axis containing the origin in a real 3D space.
    """

    _symb = Symb.ROTOREFL + Symb.INF
    _name = "infinite-fold rotoreflection axis"


class AxisRotationAxes(DirectionTransformable, InfSymmetryElement):
    """
    All two-fold rotation axes perpendicular to an infinite-fold rotation axis
    containing the origin in a real 3D space.
    """

    _symb = Symb.INF + Symb.ROT + "2"
    _name = "set of all two-fold rotation axes perpendicular to an axis"
    _id = 2


class CenterRotationAxes(InvariantTransformable, InfSymmetryElement):
    """
    All infinite-fold rotation axes containing the origin as an invariant
    center in a real 3D space.
    """

    _symb = Symb.INF + Symb.ROT + Symb.INF
    _name = "set of all infinite-fold rotation axes containing a center"


class AxisReflectionPlanes(DirectionTransformable, InfSymmetryElement):
    """
    All reflection planes containing an infinite-fold rotation axis and the
    origin in a real 3D space.
    """

    _symb = Symb.INF + Symb.REFL + "v"
    _name = "set of all reflection planes containing an axis"
    _id = -1


class CenterReflectionPlanes(InvariantTransformable, InfSymmetryElement):
    """
    All reflection planes containing the origin as an invariant center in a
    real 3D space.
    """

    _symb = Symb.INF + Symb.REFL
    _name = "set of all reflection planes containing a center"
    _id = -1


class CenterRotoreflectionAxes(InvariantTransformable, InfSymmetryElement):
    """
    All infinite-fold rotoreflection axes containing the origin as an invariant
    center in a real 3D space.
    """

    _symb = Symb.INF + Symb.ROTOREFL + Symb.INF
    _name = "set of all infinite-fold rotoreflection axes containing a center"


VectorSymmetryElement = Union[
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    AxisReflectionPlanes,
]
VEC_SYMM_ELEMS = (
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    AxisReflectionPlanes,
)
_SYMM_ELEM_ORDER = (
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    CenterRotoreflectionAxes,
    InversionCenter,
    ReflectionPlane,
    AxisReflectionPlanes,
    CenterReflectionPlanes,
    AxisRotationAxes,
    RotationAxis,
    InfRotationAxis,
    CenterRotationAxes,
)


def labeled_symm_elem(
    symm_elem: SymmetryElement, label: str
) -> SymmetryElement:
    """Return a symmetry element `symm_elem` with a label `label`."""
    symm_elem.label = label
    return symm_elem


def symm_elem_rank(prop: Tuple) -> Tuple:
    """
    Calculate the ordering rank of a symmetry element by using its properties.
    """
    rank: Tuple = (_SYMM_ELEM_ORDER.index(prop[0]),)
    if len(prop) > 1:
        rank += (prop[1],)
    return rank


class SymmetryElements:
    """
    Set of symmetry elements containing numbers of their types and of the
    angles between their axes or normals.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._included: List[VectorSymmetryElement] = []
        self._excluded: List[VectorSymmetryElement] = []
        self._types: Dict[Tuple, int] = {}
        self._angles: Dict[FrozenSet[Tuple], Dict[float, int]] = {}

    @property
    def included(self) -> Tuple[VectorSymmetryElement, ...]:
        """Included symmetry elements containing a direction."""
        return tuple(self._included)

    @property
    def excluded(self) -> Tuple[VectorSymmetryElement, ...]:
        """Excluded symmetry elements containing a direction."""
        return tuple(self._excluded)

    @property
    def types(self) -> Dict[Tuple, int]:
        """Types of symmetry elements and their numbers."""
        return self._types.copy()

    @property
    def angles(
        self,
    ) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
        """
        Angles between axes or normals of symmetry elements and their numbers.
        """
        return {props: angles.copy() for props, angles in self._angles.items()}

    def include(
        self,
        symm_elems: Union[SymmetryElement, Sequence[SymmetryElement]],
        tol: float,
    ) -> None:
        """
        Include information of one or multiple symmetry elements `symm_elems`
        using a tolerance `tol` to calculate exact intersection angles.
        """
        if not isinstance(symm_elems, Sequence):
            symm_elems = (symm_elems,)
        for symm_elem in symm_elems:
            prop = symm_elem.props
            if isinstance(symm_elem, VEC_SYMM_ELEMS):
                if prop not in self._types:
                    self._types[prop] = 0
                self._types[prop] += 1
                vec = symm_elem.vec
                for vec_symm_elem in self._included:
                    angle = intersectangle(vec, vec_symm_elem.vec)
                    for special_angle in SPECIAL_ANGLES:
                        diff = abs(angle - special_angle)
                        if diff <= tol:
                            angle = special_angle
                            break
                    else:
                        nom, denom = rational(angle / PI, tol)
                        angle = nom * PI / denom
                    if angle == 0.0 and symm_elem.similar(vec_symm_elem):
                        raise ValueError(
                            f"a parallel {symm_elem.name} already included"
                        )
                    props = frozenset((prop, vec_symm_elem.props))
                    if props not in self._angles:
                        self._angles[props] = {}
                    if angle not in self._angles[props]:
                        self._angles[props][angle] = 0
                    elif self._angles[props][angle] == 0:
                        raise ValueError(
                            f"the excluded angle of {angle} between"
                            + f" a {symm_elem.name} and a {vec_symm_elem.name}"
                            + " cannot be included"
                        )
                    self._angles[props][angle] += 1
                for vec_symm_elem in self._excluded:
                    angle = intersectangle(vec, vec_symm_elem.vec)
                    if angle > tol:
                        continue
                    angle = 0.0
                    if symm_elem.similar(vec_symm_elem):
                        raise ValueError(
                            f"the excluded parallel {symm_elem.name} cannot"
                            + " be included"
                        )
                    props = frozenset((prop, vec_symm_elem.props))
                    if props not in self._angles:
                        self._angles[props] = {}
                    if (
                        angle in self._angles[props]
                        and self._angles[props][angle] > 0
                    ):
                        raise ValueError(
                            f"the included angle of {angle} between"
                            + f" a {symm_elem.name} and a {vec_symm_elem.name}"
                            + " cannot be excluded"
                        )
                    self._angles[props][angle] = 0
                self._included.append(symm_elem)
            else:
                if prop in self._types:
                    raise ValueError(
                        f"an {symm_elem.name} already "
                        + (
                            "included"
                            if self._types[prop] == 1
                            else "excluded"
                        )
                    )
                self._types[prop] = 1

    def exclude(
        self,
        symm_elems: Union[SymmetryElement, Sequence[SymmetryElement]],
        tol: float,
    ) -> None:
        """
        Exclude information of one or multiple symmetry elements `symm_elems`
        using a tolerance `tol` to calculate exact intersection angles.
        """
        if not isinstance(symm_elems, Sequence):
            symm_elems = (symm_elems,)
        for symm_elem in symm_elems:
            prop = symm_elem.props
            if isinstance(symm_elem, VEC_SYMM_ELEMS):
                vec = symm_elem.vec
                for vec_symm_elem in self._included:
                    angle = intersectangle(vec, vec_symm_elem.vec)
                    if angle > tol:
                        continue
                    angle = 0.0
                    if symm_elem.similar(vec_symm_elem):
                        raise ValueError(
                            f"the included parallel {symm_elem.name} cannot"
                            + " be excluded"
                        )
                    props = frozenset((prop, vec_symm_elem.props))
                    if props not in self._angles:
                        self._angles[props] = {}
                    if (
                        angle in self._angles[props]
                        and self._angles[props][angle] > 0
                    ):
                        raise ValueError(
                            f"the included angle of {angle} between"
                            + f" a {symm_elem.name} and a {vec_symm_elem.name}"
                            + " cannot be excluded"
                        )
                    self._angles[props][angle] = 0
                for vec_symm_elem in self._excluded:
                    angle = intersectangle(vec, vec_symm_elem.vec)
                    if angle <= tol and symm_elem.similar(vec_symm_elem):
                        raise ValueError(
                            f"a parallel {symm_elem.name} already excluded"
                        )
                self._excluded.append(symm_elem)
            else:
                if prop in self._types:
                    raise ValueError(
                        f"an {symm_elem.name} already "
                        + (
                            "included"
                            if self._types[prop] == 1
                            else "excluded"
                        )
                    )
                self._types[prop] = 0

    def contains(self, other: "SymmetryElements") -> bool:
        """
        Check whether another instance `other` is a subset of the instance.
        """
        for key1, other_num in other._types.items():
            self_num = self._types[key1] if key1 in self._types else 0
            if self_num < other_num or (self_num == 0) != (other_num == 0):
                return False
        for key2, angles in other._angles.items():
            for angle, other_num in angles.items():
                self_num = (
                    self._angles[key2][angle]
                    if angle in self._angles[key2]
                    else 0
                )
                if self_num < other_num or (self_num == 0) != (other_num == 0):
                    return False
        return True

    @property
    def symbs(self) -> Tuple[str, ...]:
        """Sorted symbols of symmetry elements."""
        res = {}
        for prop, num in self._types.items():
            if num > 0:
                symb = prop[0]._symb
                if len(prop) > 1:
                    symb += str(prop[1])
                if num > 1:
                    symb = str(num) + symb
                res[symm_elem_rank(prop)] = symb
        return tuple(res[rank] for rank in sorted(res, reverse=True))
