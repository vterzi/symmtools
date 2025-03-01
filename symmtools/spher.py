__all__ = ["Basis", "SphericalRotation", "SphericalFunctions"]

from abc import ABC, abstractmethod
from math import sqrt
from cmath import exp
from copy import copy
from typing import Sequence, Dict, List, Tuple

from .const import PRIMAX, SPHER_FUNC_TYPES
from .linalg3d import (
    Vector,
    alignvec,
    spherfunclabels,
    complexspher,
    rotmatspher,
)
from .transform import Inversion, Rotation, Reflection, Rotoreflection
from .primitive import Points, LabeledPoint


class Basis:
    def __init__(
        self, data: Dict[str, str], order: Dict[str, Sequence[str]]
    ) -> None:
        basis = {}
        max_degree = 0
        for label, orb_types in data.items():
            degrees = []
            for func_type in orb_types.lower():
                if func_type not in SPHER_FUNC_TYPES:
                    raise ValueError(f"unknown orbital type '{func_type}'")
                if func_type not in order and func_type != SPHER_FUNC_TYPES[0]:
                    raise ValueError(
                        f"unknown order for orbital type '{func_type}'"
                    )
                degree = SPHER_FUNC_TYPES.index(func_type)
                if degree > max_degree:
                    max_degree = degree
                degrees.append(degree)
            basis[label] = tuple(degrees)
        self._basis = basis
        mapping: List[Tuple[int, ...]] = [() for _ in range(max_degree + 1)]
        mapping[0] = (0,)
        for func_type, funcs in order.items():
            func_type = func_type.lower()
            if len(func_type) != 1 or func_type not in SPHER_FUNC_TYPES:
                raise ValueError(f"unknown orbital type '{func_type}'")
            degree = SPHER_FUNC_TYPES.index(func_type)
            if degree > max_degree:
                continue
            idxs = []
            for func in spherfunclabels(degree):
                if func not in funcs:
                    raise ValueError(
                        f"missing function '{func}' for orbital type"
                        + f" '{func_type}'"
                    )
                idxs.append(funcs.index(func))
            mapping[degree] = tuple(idxs)
        self._order = tuple(mapping)


class SphericalFunctions:
    def __init__(
        self, basis: Basis, points: Points, coefs: Sequence[float]
    ) -> None:
        n_all_coefs = len(coefs)
        all_coefs_arr = []
        i_coef = 0
        for point in points:
            if not isinstance(point, LabeledPoint):
                raise ValueError("not a `LabeledPoint` instance")
            coefs_arr = []
            label = point.label
            if label not in basis._basis:
                raise ValueError(f"no basis set for '{label}'")
            for degree in basis._basis[label]:
                order = basis._order[degree]
                new_i_coef = i_coef + len(order)
                if new_i_coef > n_all_coefs:
                    raise ValueError("wrong number of coefficients")
                coefs_arr.append(
                    tuple(
                        complexspher(
                            tuple(coefs[i_coef + idx] for idx in order)
                        )
                    )
                )
                i_coef = new_i_coef
            all_coefs_arr.append(tuple(coefs_arr))
        if i_coef != n_all_coefs:
            raise ValueError("wrong number of coefficients")
        self._points = points
        self._coefs = tuple(all_coefs_arr)

    def invert(self) -> "SphericalFunctions":
        res = copy(self)
        res._points = Inversion()(res._points)
        all_coefs_arr = []
        for old_coefs_arr in res._coefs:
            new_coefs_arr = []
            for old_coefs in old_coefs_arr:
                degree = len(old_coefs) // 2
                if degree % 2 == 1:
                    new_coefs = tuple(-old_coef for old_coef in old_coefs)
                else:
                    new_coefs = old_coefs
                new_coefs_arr.append(new_coefs)
            all_coefs_arr.append(tuple(new_coefs_arr))
        res._coefs = tuple(all_coefs_arr)
        return res

    def rotate(self, rot: "SphericalRotation") -> "SphericalFunctions":
        res = copy(self)
        res._points = rot._rot(res._points)
        all_coefs_arr = []
        for old_coefs_arr in res._coefs:
            new_coefs_arr = []
            for old_coefs in old_coefs_arr:
                degree = len(old_coefs) // 2
                mat = rot[degree]
                new_coefs = []
                for row in mat:
                    new_coef = 0.0j
                    for entry, old_coef in zip(row, old_coefs):
                        new_coef += entry * old_coef
                    new_coefs.append(new_coef)
                new_coefs_arr.append(tuple(new_coefs))
            all_coefs_arr.append(tuple(new_coefs_arr))
        res._coefs = tuple(all_coefs_arr)
        return res

    def align(self, vec: Vector) -> "SphericalFunctions":
        degrees = set()
        for coefs_arr in self._coefs:
            for coefs in coefs_arr:
                degrees.add(len(coefs) // 2)
        return self.rotate(
            SphericalRotation(
                tuple(sorted(degrees)),
                Rotation.from_quat(alignvec(PRIMAX, vec)),
            )
        )

    def orthrotate(self, angle: float) -> "SphericalFunctions":
        res = copy(self)
        res._points = Rotation(PRIMAX, angle)(res._points)
        all_coefs_arr = []
        for old_coefs_arr in res._coefs:
            new_coefs_arr = []
            for old_coefs in old_coefs_arr:
                degree = len(old_coefs) // 2
                new_coefs = []
                for i, old_coef in enumerate(old_coefs):
                    new_coefs.append(
                        old_coef * exp(complex(0.0, (degree - i) * angle))
                    )
                new_coefs_arr.append(tuple(new_coefs))
            all_coefs_arr.append(tuple(new_coefs_arr))
        res._coefs = tuple(all_coefs_arr)
        return res

    def reflect(self) -> "SphericalFunctions":
        res = copy(self)
        res._points = Reflection(PRIMAX)(res._points)
        all_coefs_arr = []
        for old_coefs_arr in res._coefs:
            new_coefs_arr = []
            for old_coefs in old_coefs_arr:
                new_coefs = tuple(
                    old_coef if i % 2 == 0 else -old_coef
                    for i, old_coef in enumerate(old_coefs)
                )
                new_coefs_arr.append(new_coefs)
            all_coefs_arr.append(tuple(new_coefs_arr))
        res._coefs = tuple(all_coefs_arr)
        return res

    def chars(
        self, transform: "SphericalTransformation", tol: float
    ) -> List[complex]:
        points = self._points
        all_coefs_arr = self._coefs

        def vector(obj: "SphericalFunctions") -> List[complex]:
            all_coefs_arr = obj._coefs
            vec: List[complex] = []
            for i in points.order(obj._points):
                for coefs in all_coefs_arr[i]:
                    vec.extend(coefs)
            return vec

        def spherfuncs(vec: List[complex]) -> "SphericalFunctions":
            res = copy(self)
            i = 0
            new_all_coefs_arr = []
            for old_coefs_arr in all_coefs_arr:
                new_coefs_arr = []
                for old_coefs in old_coefs_arr:
                    new_coefs = []
                    for _ in range(len(old_coefs)):
                        new_coefs.append(vec[i])
                        i += 1
                    new_coefs_arr.append(tuple(new_coefs))
                new_all_coefs_arr.append(tuple(new_coefs_arr))
            res._coefs = tuple(new_all_coefs_arr)
            return res

        def sqnorm(vec: List[complex]) -> float:
            return sum((coef.conjugate() * coef).real for coef in vec)

        def norm(vec: List[complex]) -> float:
            return sqrt(sqnorm(vec))

        def mul(vec: List[complex], scalar: complex) -> None:
            for i in range(len(vec)):
                vec[i] *= scalar

        def normalize(vec: List[complex]) -> None:
            mul(vec, 1.0 / norm(vec))

        def dot(vec1: List[complex], vec2: List[complex]) -> complex:
            return sum(
                coef1.conjugate() * coef2 for coef1, coef2 in zip(vec1, vec2)
            )

        chars = []
        vecs: List[List[complex]] = []
        obj = transform.align(self)
        vec = vector(obj)
        normalize(vec)
        while True:
            vecs.append(vec)
            new_vec = vector(transform.apply(obj))
            normalize(new_vec)
            chars.append(
                dot(vec, new_vec) / sqrt(sqnorm(vec) * sqnorm(new_vec))
            )
            vec = new_vec
            for unitvec in vecs:
                scalar = dot(vec, unitvec)
                for i, coef in enumerate(unitvec):
                    vec[i] -= scalar * coef
            norm_vec = norm(vec)
            if norm_vec > tol:
                mul(vec, 1.0 / norm_vec)
                obj = spherfuncs(vec)
            else:
                break
        return chars


class SphericalTransformation(ABC):
    @abstractmethod
    def align(self, funcs: SphericalFunctions) -> SphericalFunctions:
        pass

    @abstractmethod
    def apply(self, funcs: SphericalFunctions) -> SphericalFunctions:
        pass


class SphericalInversion(SphericalTransformation):
    def align(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs

    def apply(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs.invert()


class SphericalRotation(SphericalTransformation):
    def __init__(self, degrees: Sequence[int], rot: Rotation) -> None:
        degrees = tuple(sorted(set(degrees)))
        for degree in degrees:
            if degree < 0:
                raise ValueError("negative degree")
        self._degrees = degrees
        mats = rotmatspher(degrees, rot.quat)
        self._rot = rot
        self._mats = tuple(tuple(tuple(row) for row in mat) for mat in mats)

    def __getitem__(self, degree: int) -> Tuple[Tuple[complex, ...], ...]:
        degrees = self._degrees
        if degree not in degrees:
            raise ValueError(f"no rotation matrix for degree {degree}")
        return self._mats[degrees.index(degree)]

    def align(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs

    def apply(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs.rotate(self)


class SphericalReflection(SphericalTransformation):
    def __init__(self, refl: Reflection) -> None:
        self._refl = refl

    def align(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs.align(self._refl.vec)

    def apply(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs.reflect()


class SphericalRotoreflection(SphericalTransformation):
    def __init__(self, rotorefl: Rotoreflection) -> None:
        self._rotorefl = rotorefl

    def align(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs.align(self._rotorefl.vec)

    def apply(self, funcs: SphericalFunctions) -> SphericalFunctions:
        return funcs.orthrotate(self._rotorefl.angle).reflect()
