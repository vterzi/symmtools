__all__ = ('Point', 'LabeledPoint', 'Arrow', 'Elems', 'Struct')

from abc import ABC, abstractmethod
from copy import copy

from numpy import inf, empty, zeros, sign, dot
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

from .vecop import vector, normalize, translate, invert, translate2, reflect
from .transform import Identity, Translation, Inversion, Rotation, Reflection, Rotoreflection


class Primitive(ABC):
    def args(self):
        return ''

    def __str__(self):
        return f'{self.__class__.__name__}({self.args()})'

    def __repr__(self):
        return self.__str__()

    def diff(self, primitive):
        return 0 if type(self) == type(primitive) else inf

    def same(self, primitive, tol):
        return self.diff(primitive) <= tol

    def copy(self):
        return copy(self)

    def transform(self, transform):
        type_transform = type(transform)
        if type_transform == Identity:
            return self.copy()
        elif type_transform == Translation:
            return self.translate(transform)
        elif type_transform == Inversion:
            return self.invert()
        elif type_transform == Rotation:
            return self.rotate(transform)
        elif type_transform == Reflection:
            return self.reflect(transform)
        elif type_transform == Rotoreflection:
            return self.rotoreflect(transform)
        else:
            raise ValueError(f'illegal transformation: {type_transform}')

    @abstractmethod
    def translate(self, translation):
        pass

    @abstractmethod
    def invert(self):
        pass

    @abstractmethod
    def rotate(self, rotation):
        pass

    @abstractmethod
    def reflect(self, reflection):
        pass

    @abstractmethod
    def rotoreflect(self, rotoreflection):
        pass


class Point(Primitive):
    def __init__(self, pos):
        self._pos = vector(pos)

    def args(self):
        return str(list(self._pos)).replace(' ', '')

    @property
    def pos(self):
        return self._pos

    def diff(self, point):
        diff = super().diff(point)
        if diff < inf:
            diff = max(diff, norm(self._pos - point.pos))
        return diff

    def translate(self, translation):
        res = self.copy()
        res._pos = translate(self._pos, translation.vec)
        return res

    def invert(self):
        res = self.copy()
        res._pos = invert(self._pos)
        return res

    def rotate(self, rotation):
        res = self.copy()
        res._pos = translate2(self._pos, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(self, reflection):
        res = self.copy()
        res._pos = reflect(self._pos, reflection.vec)
        return res

    def rotoreflect(self, rotoreflection):
        res = self.copy()
        res._pos = reflect(translate2(self._pos, rotoreflection.vec, rotoreflection.cos, rotoreflection.sin),
                           rotoreflection.vec)
        return res


class LabeledPoint(Point):
    def __init__(self, pos, label):
        super().__init__(pos)
        self._label = label

    def args(self):
        label = self._label.replace('"', '\\"')
        return f'{super().args()},{label}'

    @property
    def label(self):
        return self._label

    def diff(self, point):
        diff = super().diff(point)
        if diff < inf:
            diff = max(diff, 0 if self._label == point.label else inf)
        return diff


class Arrow(Primitive):
    def __init__(self, vec, fore, back):
        self._vec = normalize(vector(vec))
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
        vec = str(list(self._vec)).replace(' ', '')
        return f'{vec},{fore},{back}'

    @property
    def vec(self):
        return self._vec

    @property
    def phase(self):
        return self._phase

    def diff(self, arrow):
        diff = super().diff(arrow)
        if diff < inf:
            vec = arrow.vec if dot(self._vec, arrow.vec) >= 0 else - arrow.vec
            diff = max(diff, norm(self._vec - vec), (0 if abs(self._phase) == abs(arrow.phase) else inf))
        return diff

    def same(self, arrow, tol):
        return ((int(sign(dot(self._vec, arrow.vec))) if self._phase == 0 else self._phase * arrow.phase)
                if self.diff(arrow) <= tol else 0)

    def translate(self, translation):
        res = self.copy()
        res._vec = self._vec.copy()
        return res

    def invert(self):
        res = self.copy()
        res._vec = invert(self._vec)
        return res

    def rotate(self, rotation):
        res = self.copy()
        res._vec = translate2(self._vec, rotation.vec, rotation.cos, rotation.sin)
        return res

    def reflect(self, reflection):
        res = self.copy()
        res._vec = reflect(self._vec, reflection.vec)
        return res

    def rotoreflect(self, rotoreflection):
        res = self.copy()
        res._vec = reflect(translate2(self._vec, rotoreflection.vec, rotoreflection.cos, rotoreflection.sin),
                           rotoreflection.vec)
        return res


class Elems(Primitive):
    def __init__(self, elems):
        self._elems = tuple(elems)

    def args(self):
        return ('['
                + ('' if not self._elems
                   else '\n  ' + ',\n  '.join([str(elem).replace('\n', '\n  ') for elem in self._elems]) + ',\n')
                + ']')

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
            raise ValueError(f'different number of elements in the instances of {self.__class__.__name__}')
        diffs = empty((n, n))
        for i1 in range(n):
            elem = self._elems[i1]
            for i2 in range(n):
                diffs[i1, i2] = elem.diff(elems.elems[i2])
        try:
            order = linear_sum_assignment(diffs)[1]
        except ValueError:
            raise ValueError(f'the instances of {self.__class__.__name__} differ')
        elems = n * [None]
        for i in range(n):
            elems[order[i]] = self._elems[i]
        self._elems = tuple(elems)
        diff = 0
        for i in range(n):
            diff = max(diff, diffs[i, order[i]])
        return diff

    def diff(self, elems):
        diff = super().diff(elems)
        if diff < inf:
            try:
                diff = max(diff, elems.sort(self))
            except ValueError:
                diff = inf
        return diff

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
    def pos(self):
        centroid = zeros(3)
        for elem in self._elems:
            centroid += elem.pos
        n = len(self._elems)
        if n > 0:
            centroid /= n
        return centroid

    def center(self):
        return self.translate(Translation(- self.pos))

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
    def __init__(self, pos, coef=1, arrows=()):
        Point.__init__(self, pos)
        Elems.__init__(self, arrows)
        self._coef = abs(coef)
        self._phase = int(sign(coef))

    def args(self):
        return f'{Point.args(self)},{self._phase * self._coef},{Elems.args(self)}'

    @property
    def coef(self):
        return self._coef

    @property
    def phase(self):
        return self._phase

    def diff(self, struct):
        diff = max(Point.diff(self, struct), Elems.diff(self, struct))
        if diff < inf:
            diff = max(diff, abs(self._coef - struct.coef))
        return diff

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
        return Elems.rotoreflect(Point.rotoreflect(self, rotoreflection), rotoreflection)
