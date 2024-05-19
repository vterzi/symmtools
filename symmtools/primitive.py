"""Classes for primitive objects in a real 3D space."""

__all__ = ["Point", "LabeledPoint", "Arrow", "Elems", "Struct"]

from numpy import empty, zeros, sign
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

from .const import INF
from .transform import (
    Transformable,
    VecTransformable,
    DirectionTransformable,
    Translation,
)


class Point(VecTransformable):
    """Point in a real 3D space."""

    pass


class LabeledPoint(Point):
    def __init__(self, vec, label):
        super().__init__(vec)
        self._label = label

    def args(self):
        label = self._label.replace('"', '\\"')
        return f"{super().args()},{label}"

    @property
    def label(self):
        return self._label

    def diff(self, point):
        res = super().diff(point)
        if res < INF and self._label != point.label:
            res = INF
        return res


class Arrow(DirectionTransformable):
    def __init__(self, vec, fore, back):
        super().__init__(vec)
        if fore and back:
            self._phase = 1
        elif not fore and not back:
            self._phase = -1
        else:
            self._phase = 0
            if back:
                self._vec *= -1

    def args(self):
        if self._phase == 1:
            fore = True
            back = True
        elif self._phase == -1:
            fore = False
            back = False
        else:
            fore = True
            back = False
        return f"{super().args()},{fore},{back}"

    @property
    def phase(self):
        return self._phase

    def diff(self, arrow):
        res = super().diff(arrow)
        if res < INF:
            vec = arrow.vec if self._vec.dot(arrow.vec) >= 0 else -arrow.vec
            res = max(
                res,
                norm(self._vec - vec),
                (0 if abs(self._phase) == abs(arrow.phase) else INF),
            )
        return res

    def same(self, arrow, tol):
        return (
            (
                int(sign(self._vec.dot(arrow.vec)))
                if self._phase == 0
                else self._phase * arrow.phase
            )
            if self.diff(arrow) <= tol
            else 0
        )


class Elems(Transformable):
    def __init__(self, elems):
        self._elems = tuple(elems)

    def args(self):
        return (
            "["
            + (
                ""
                if not self._elems
                else "\n  "
                + ",\n  ".join(
                    [str(elem).replace("\n", "\n  ") for elem in self._elems]
                )
                + ",\n"
            )
            + "]"
        )

    def __getitem__(self, item):
        return self._elems[item]

    def __len__(self):
        return len(self._elems)

    @property
    def elems(self):
        return self._elems

    def sort(self, elems):
        n = len(self._elems)
        if n != len(elems.elems):
            raise ValueError(
                f"different number of elements in the instances of {self.__class__.__name__}"
            )
        diffs = empty((n, n))
        for i1 in range(n):
            elem = self._elems[i1]
            for i2 in range(n):
                diffs[i1, i2] = elem.diff(elems.elems[i2])
        try:
            order = linear_sum_assignment(diffs)[1]
        except ValueError:
            raise ValueError(
                f"the instances of {self.__class__.__name__} differ"
            )
        elems = n * [None]
        for i in range(n):
            elems[order[i]] = self._elems[i]
        self._elems = tuple(elems)
        diff = 0
        for i in range(n):
            diff = max(diff, diffs[i, order[i]])
        return diff

    def diff(self, elems):
        res = super().diff(elems)
        if res < INF:
            try:
                res = max(res, elems.sort(self))
            except ValueError:
                res = INF
        return res

    def same(self, elems, tol):
        def add(parity1, parity2):
            bool1 = isinstance(parity1, bool)
            bool2 = isinstance(parity2, bool)
            if bool1 and bool2:
                return parity1 and parity2
            elif bool1 or bool2:
                return parity1 * parity2
            else:
                return parity1 if parity1 == parity2 else 0

        symm = self.diff(elems) <= tol
        for i in range(len(self._elems)):
            if not symm:
                break
            symm = add(symm, self._elems[i].same(elems.elems[i], tol))
        return symm

    def check(self, tol):
        n = len(self._elems)
        for i1 in range(n - 1):
            elem = self._elems[i1]
            for i2 in range(i1 + 1, n):
                if elem.diff(self._elems[i2]) <= tol:
                    return False
        return True

    def translate(self, translation):
        res = self.copy()
        elems = []
        for elem in self._elems:
            elems.append(elem.translate(translation))
        res._elems = tuple(elems)
        return res

    @property
    def vec(self):
        centroid = zeros(3)
        for elem in self._elems:
            centroid += elem.vec
        n = len(self._elems)
        if n > 0:
            centroid /= n
        return centroid

    def center(self):
        return self.translate(Translation(-self.vec))

    def invert(self):
        res = self.copy()
        elems = []
        for elem in self._elems:
            elems.append(elem.invert())
        res._elems = tuple(elems)
        return res

    def rotate(self, rotation):
        res = self.copy()
        elems = []
        for elem in self._elems:
            elems.append(elem.rotate(rotation))
        res._elems = tuple(elems)
        return res

    def reflect(self, reflection):
        res = self.copy()
        elems = []
        for elem in self._elems:
            elems.append(elem.reflect(reflection))
        res._elems = tuple(elems)
        return res

    def rotoreflect(self, rotoreflection):
        res = self.copy()
        elems = []
        for elem in self._elems:
            elems.append(elem.rotoreflect(rotoreflection))
        res._elems = tuple(elems)
        return res


class Struct(Point, Elems):
    def __init__(self, vec, coef=1, arrows=()):
        Point.__init__(self, vec)
        Elems.__init__(self, arrows)
        self._coef = abs(coef)
        self._phase = int(sign(coef))

    def args(self):
        return (
            f"{Point.args(self)},{self._phase * self._coef},{Elems.args(self)}"
        )

    @property
    def coef(self):
        return self._coef

    @property
    def phase(self):
        return self._phase

    def diff(self, struct):
        res = max(Point.diff(self, struct), Elems.diff(self, struct))
        if res < INF:
            res = max(res, abs(self._coef - struct.coef))
        return res

    def same(self, struct, tol):
        return self._phase * struct.phase * Elems.same(self, struct, tol)

    def translate(self, translation):
        return Elems.translate(Point.translate(self, translation), translation)

    def center(self):
        pass

    def invert(self):
        return Elems.invert(Point.invert(self))

    def rotate(self, rotation):
        return Elems.rotate(Point.rotate(self, rotation), rotation)

    def reflect(self, reflection):
        return Elems.reflect(Point.reflect(self, reflection), reflection)

    def rotoreflect(self, rotoreflection):
        return Elems.rotoreflect(
            Point.rotoreflect(self, rotoreflection), rotoreflection
        )
