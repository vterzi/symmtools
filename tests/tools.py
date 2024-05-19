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
    "eye",
    "sin",
    "cos",
    "cross",
    "norm",
    "ndarray",
    "float64",
    "NDArray",
    "randsign",
    "randvec",
    "randne0vec",
    "randunitvec",
    "perturb",
    "orthperturb",
    "randangle",
    "randne0angle",
]

from random import random, choice, randrange, randint, normalvariate
from typing import Tuple

from numpy import pi, array, zeros, eye, sin, cos, cross, ndarray, float64
from numpy.linalg import norm
from numpy.typing import NDArray

from symmtools import TOL


def randsign() -> int:
    return choice([-1, 1])


def randvec() -> NDArray[float64]:
    return array([normalvariate(0, 1) for _ in range(3)])


def randne0vec() -> NDArray[float64]:
    vec_norm = float64(0)
    while vec_norm == 0:
        vec = randvec()
        vec_norm = norm(vec)
    return vec


def randunitvec() -> NDArray[float64]:
    vec = randne0vec()
    return vec / norm(vec)


def perturb() -> NDArray[float64]:
    vec = zeros(3)
    vec[randrange(3)] = randsign() * TOL
    return vec


def orthperturb(unitvec: NDArray[float64]) -> NDArray[float64]:
    vec_norm = float64(0)
    while vec_norm == 0:
        vec = randvec()
        vec -= vec.dot(unitvec) * unitvec
        vec_norm = norm(vec)
    return vec * TOL / vec_norm


def randangle() -> float:
    return 2 * pi * random()


def randne0angle() -> float:
    angle = 0.0
    while angle == 0:
        angle = 2 * pi * random()
    return angle
