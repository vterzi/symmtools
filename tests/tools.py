__all__ = [
    "random",
    "choice",
    "randrange",
    "randint",
    "normalvariate",
    "Tuple",
    "pi",
    "array",
    "zeros",
    "sin",
    "cos",
    "cross",
    "ndarray",
    "float64",
    "NDArray",
    "randvec",
    "randnonzerovec",
    "randunitvec",
    "perturb",
]

from random import random, choice, randrange, randint, normalvariate
from typing import Tuple

from numpy import pi, array, zeros, sin, cos, cross, ndarray, float64
from numpy.typing import NDArray

from symmtools import TOL


def randvec() -> NDArray[float64]:
    return array([normalvariate(0, 1) for _ in range(3)])


def randnonzerovec() -> Tuple[NDArray[float64], float64]:
    norm = float64(0)
    while norm == 0:
        vec = randvec()
        norm = (vec**2).sum() ** 0.5
    return vec, norm


def randunitvec() -> NDArray[float64]:
    vec, norm = randnonzerovec()
    return vec / norm


def perturb() -> NDArray[float64]:
    vec = zeros(3)
    vec[randrange(3)] = choice([-1, 1]) * TOL
    return vec