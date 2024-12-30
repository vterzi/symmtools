__all__ = [
    "TestCase",
    "main",
    "pi",
    "sqrt",
    "cos",
    "sin",
    "acos",
    "random",
    "choice",
    "randint",
    "randrange",
    "normalvariate",
    "Union",
    "Sequence",
    "Tuple",
    "List",
    "array",
    "empty",
    "zeros",
    "eye",
    "clip",
    "cross",
    "roots",
    "ndarray",
    "float64",
    "norm",
    "linear_sum_assignment",
    "EPS",
    "TOL",
    "vec3D",
    "randsign",
    "randfloat",
    "randangle",
    "randvec",
    "randunitvec",
    "randmat",
    "perturb",
    "orthperturb",
]

from unittest import TestCase, main
from math import pi, sqrt, cos, sin, acos
from random import random, choice, randint, randrange, normalvariate
from typing import Union, Sequence, Tuple, List

from numpy import (
    array,
    empty,
    zeros,
    eye,
    clip,
    cross,
    roots,
    ndarray,
    float64,
)
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # type: ignore

EPS = 7.0 / 3.0 - 4.0 / 3.0 - 1.0
TOL = 2**10 * EPS


def vec3D(vec: NDArray[float64]) -> Tuple[float, float, float]:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def randsign() -> int:
    return choice((-1, 1))


def randfloat() -> float:
    return normalvariate(0.0, 1.0)


def randangle(nonzero: bool = False) -> float:
    angle = 0.0
    while angle == 0.0:
        angle = 2.0 * pi * random()
        if not nonzero:
            break
    return angle


def randvec(nonzero: bool = False) -> Tuple[float, float, float]:
    vec_sq_norm = 0.0
    while vec_sq_norm == 0.0:
        x = randfloat()
        y = randfloat()
        z = randfloat()
        if not nonzero:
            break
        vec_sq_norm = x * x + y * y + z * z
    return (x, y, z)


def randunitvec() -> Tuple[float, float, float]:
    vec_sq_norm = 0.0
    while vec_sq_norm == 0.0:
        x = randfloat()
        y = randfloat()
        z = randfloat()
        vec_sq_norm = x * x + y * y + z * z
    scalar = 1.0 / sqrt(vec_sq_norm)
    return (x * scalar, y * scalar, z * scalar)


def randmat() -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    return (randvec(), randvec(), randvec())


def perturb() -> Tuple[float, float, float]:
    vec = [0.0, 0.0, 0.0]
    vec[randrange(3)] = randsign() * TOL
    return (vec[0], vec[1], vec[2])


def orthperturb(
    unitvec: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    unitvec_x = unitvec[0]
    unitvec_y = unitvec[1]
    unitvec_z = unitvec[2]
    vec_sq_norm = 0.0
    while vec_sq_norm == 0.0:
        vec_x = randfloat()
        vec_y = randfloat()
        vec_z = randfloat()
        prod = vec_x * unitvec_x + vec_y * unitvec_y + vec_z * unitvec_z
        vec_x -= prod * unitvec_x
        vec_y -= prod * unitvec_y
        vec_z -= prod * unitvec_z
        vec_sq_norm = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z
    scalar = TOL / sqrt(vec_sq_norm)
    return (vec_x * scalar, vec_y * scalar, vec_z * scalar)
