__all__ = [
    "TestCase",
    "main",
    "random",
    "choice",
    "randrange",
    "randint",
    "normalvariate",
    "Union",
    "Sequence",
    "Tuple",
    "List",
    "pi",
    "sqrt",
    "sin",
    "cos",
    "acos",
    "clip",
    "array",
    "zeros",
    "eye",
    "cross",
    "roots",
    "norm",
    "ndarray",
    "float64",
    "NDArray",
    "randsign",
    "randangle",
    "randvec",
    "randunitvec",
    "perturb",
    "orthperturb",
]

from unittest import TestCase, main
from math import pi, sqrt, sin, cos, acos
from random import random, choice, randrange, randint, normalvariate
from typing import Union, Sequence, Tuple, List

from numpy import (
    clip,
    empty,
    array,
    zeros,
    eye,
    cross,
    roots,
    ndarray,
    float64,
)
from numpy.linalg import norm
from numpy.typing import NDArray

from symmtools import TOL


def randsign() -> int:
    return choice((-1, 1))


def randangle(nonzero: bool = False) -> float:
    angle = 0.0
    while angle == 0.0:
        angle = 2.0 * pi * random()
        if not nonzero:
            break
    return angle


def randvec(nonzero: bool = False) -> NDArray[float64]:
    vec = empty(3)
    vec_sq_norm = 0.0
    while vec_sq_norm == 0.0:
        for i in range(3):
            vec[i] = normalvariate(0.0, 1.0)
        if not nonzero:
            break
        vec_sq_norm = vec.dot(vec)
    return vec


def randunitvec() -> NDArray[float64]:
    vec = empty(3)
    vec_sq_norm = 0.0
    while vec_sq_norm == 0.0:
        for i in range(3):
            vec[i] = normalvariate(0.0, 1.0)
        vec_sq_norm = vec.dot(vec)
    return vec / sqrt(vec_sq_norm)


def perturb() -> NDArray[float64]:
    vec = zeros(3)
    vec[randrange(3)] = randsign() * TOL
    return vec


def orthperturb(unitvec: NDArray[float64]) -> NDArray[float64]:
    vec = empty(3)
    vec_sq_norm = 0.0
    while vec_sq_norm == 0.0:
        vec = randvec()
        vec -= vec.dot(unitvec) * unitvec
        vec_sq_norm = vec.dot(vec)
    return vec * TOL / sqrt(vec_sq_norm)
