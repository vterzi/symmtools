"""Classes for primitive objects in a real 3D space."""

__all__ = ["Point", "LabeledPoint", "Arrow", "Elems", "Struct"]

from copy import copy

from numpy import empty, zeros
from scipy.optimize import linear_sum_assignment

from .const import INF
from .vecop import diff, indepunit, translate, invert, move2, reflect
from .transform import (
    Transformable,
    PosTransformable,
    DirectionTransformable,
    Translation,
    Rotation,
    Reflection,
    Rotoreflection,
)
from .typehints import Any, Sequence, Bool, RealVector


class Point(PosTransformable):
    """Point in a real 3D space."""

    pass


class LabeledPoint(Point):
    """Labeled point in a real 3D space."""

    def __init__(self, vec: RealVector, label: str) -> None:
        """
        Initialize the instance with a 3D vector `vec` and a label `label`.
        """
        super().__init__(vec)
        self._label = label

    @property
    def label(self) -> str:
        """Return the label."""
        return self._label

    def args(self):
        label = self._label.replace('"', '\\"')
        return f"{super().args()},{label}"

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF and self._label != obj.label:
            res = INF
        return res


class Arrow(DirectionTransformable):
    """Arrow in a real 3D space."""

    def __init__(self, vec: RealVector, fore: Bool, back: Bool) -> None:
        """
        Initialize the instance with a 3D non-zero vector `vec` and two
        arguments, `fore` and `back`, describing the types of its ends: `True`
        for head and `False` for tail. The direction of the vector corresponds
        the direction from the end described by `back` to the end described by
        `fore`. Possible combinations of `fore` and `back` are:
        - '<->' (two heads) for `True` and `True`,
        - '>->' (one head and one tail) for `True` and `False`,
        - '<-<' (one tail and one head) for `False` and `True`,
        - '>-<' (two tails) for `False` and `False`.
        """
        super().__init__(vec)
        if fore and back:
            self._form = 1
        elif not fore and not back:
            self._form = -1
        else:
            self._form = 0
            if back:
                self._vec = -self._vec

    @property
    def form(self) -> int:
        """
        Return the form: `1` for two heads ('<->'), `-1` for two tails ('>-<'),
        and `0` otherwise ('>->' or '<-<').
        """
        return self._form

    def args(self) -> str:
        if self._form == 1:
            fore = True
            back = True
        elif self._form == -1:
            fore = False
            back = False
        else:
            fore = True
            back = False
        return f"{super().args()},{fore},{back}"

    def diff(self, obj: Any) -> float:
        res = Transformable.diff(self, obj)
        if res < INF:
            if self._form != obj.form:
                res = INF
            elif self._form == 0:
                res = max(res, diff(self._vec, obj.vec))
            else:
                res = max(res, indepunit(self._vec, obj.vec))
        return res

    def negate(self) -> "Arrow":
        res = copy(self)
        if self._form == 0:
            res._vec = -self._vec
        else:
            res._form = -self._form
        return res


class Elems(PosTransformable):
    """Set of elements."""

    _elems: Sequence[Transformable] = ()

    def __init__(self, elems: Sequence[Transformable]) -> None:
        """Initialize the instance with a set of elements `elems`."""
        self._elems = tuple(elems)
        centroid = zeros(3)
        n = 0
        for elem in self._elems:
            if isinstance(elem, PosTransformable):
                centroid += elem.pos
                n += 1
        if n > 0:
            centroid /= n
        super().__init__(centroid)

    @property
    def elems(self) -> Sequence[Transformable]:
        """Return the set of elements."""
        return self._elems

    def args(self) -> str:
        return (
            "["
            + (
                ""
                if not self._elems
                else "\n  "
                + ",\n  ".join(
                    str(elem).replace("\n", "\n  ") for elem in self._elems
                )
                + ",\n"
            )
            + "]"
        )

    def __getitem__(self, item: int) -> Transformable:
        return self._elems[item]

    def __len__(self) -> int:
        return len(self._elems)

    def sort(self, obj: "Elems") -> float:
        """
        Sort the elements of the instance minimizing the difference to another
        instance `obj` and return the resulting difference.
        """
        n = len(self._elems)
        if n != len(obj.elems):
            raise ValueError("different number of elements")
        diffs = empty((n, n))
        for i1 in range(n):
            elem = self._elems[i1]
            for i2 in range(n):
                diffs[i1, i2] = elem.diff(obj.elems[i2])
        try:
            order = linear_sum_assignment(diffs)[1]
        except ValueError:
            raise ValueError("different number of elements of the same type")
        idxs = n * [n]
        for i in range(n):
            idxs[order[i]] = i
        self._elems = tuple(self._elems[idxs[i]] for i in range(n))
        diff = 0.0
        for i in range(n):
            diff = max(diff, diffs[i, order[i]])
        return diff

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            try:
                res = max(res, obj.sort(self))
            except ValueError:
                res = INF
        return res

    def negate(self) -> "Elems":
        res = copy(self)
        res._elems = tuple(elem.negate() for elem in self._elems)
        return res

    def nondegen(self, tol: float) -> bool:
        """
        Check wether no two elements are the same within a tolerance `tol`.
        """
        n = len(self._elems)
        for i1 in range(n - 1):
            elem = self._elems[i1]
            for i2 in range(i1 + 1, n):
                if elem.same(self._elems[i2], tol):
                    return False
        return True

    def center(self):
        return self.translate(Translation(-self.pos))

    def translate(self, translation: Translation) -> "Elems":
        return type(self)(
            tuple(elem.translate(translation) for elem in self._elems)
        )

    def invert(self) -> "Elems":
        return type(self)(tuple(elem.invert() for elem in self._elems))

    def rotate(self, rotation: Rotation) -> "Elems":
        return type(self)(tuple(elem.rotate(rotation) for elem in self._elems))

    def reflect(self, reflection: Reflection) -> "Elems":
        return type(self)(
            tuple(elem.reflect(reflection) for elem in self._elems)
        )

    def rotoreflect(self, rotoreflection: Rotoreflection) -> "Elems":
        return type(self)(
            tuple(elem.rotoreflect(rotoreflection) for elem in self._elems)
        )


class Struct(Elems):
    """Set of arrows with a coefficient."""

    def __init__(
        self, vec: RealVector, coef: float = 1.0, arrows: Sequence[Arrow] = ()
    ) -> None:
        """
        Initialize the instance with a 3D position vector `vec`, a coefficient
        `coef`, and a set of arrows `arrows`.
        """
        super().__init__(arrows)
        PosTransformable.__init__(self, vec)
        self._coef = coef

    @property
    def coef(self) -> float:
        """Return the coefficient."""
        return self._coef

    def args(self) -> str:
        return f"{PosTransformable.args(self)},{self._coef},{super().args()}"

    def diff(self, obj: Any) -> float:
        res = super().diff(obj)
        if res < INF:
            res = max(res, abs(self._coef - obj.coef))
        return res

    def negate(self) -> "Struct":
        res = copy(self)
        res._coef = -self._coef
        return res

    def translate(self, translation: Translation) -> "Struct":
        res = copy(self)
        res._vec = translate(self._vec, translation.vec)
        return res

    def invert(self) -> "Struct":
        res = copy(self)
        res._vec = invert(self._vec)
        res._elems = tuple(elem.invert() for elem in self._elems)
        return res

    def rotate(self, rotation: Rotation) -> "Struct":
        res = copy(self)
        res._vec = move2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        res._elems = tuple(elem.rotate(rotation) for elem in self._elems)
        return res

    def reflect(self, reflection: Reflection) -> "Struct":
        res = copy(self)
        res._vec = reflect(self._vec, reflection.vec)
        res._elems = tuple(elem.reflect(reflection) for elem in self._elems)
        return res

    def rotoreflect(self, rotoreflection: Rotoreflection) -> "Struct":
        res = copy(self)
        res._vec = reflect(
            move2(
                self._vec,
                rotoreflection.vec,
                rotoreflection.cos,
                rotoreflection.sin,
            ),
            rotoreflection.vec,
        )
        res._elems = tuple(
            elem.rotoreflect(rotoreflection) for elem in self._elems
        )
        return res
