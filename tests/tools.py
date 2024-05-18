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
    "ndarray",
    "float64",
    "NDArray",
    "randvec",
    "randnonzerovec",
    "randunitvec",
    "perturb",
    "randangle",
    "rotmat",
    "reflmat",
]

from random import random, choice, randrange, randint, normalvariate
from typing import Tuple

from numpy import pi, array, zeros, eye, sin, cos, cross, ndarray, float64
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


def randangle() -> float:
    return 2 * pi * random()


def rotmat(vec: NDArray[float64], angle: float) -> NDArray[float64]:
    x, y, z = vec
    c = cos(angle)
    s = sin(angle)
    return array(
        [
            [
                c + x * x * (1 - c),
                x * y * (1 - c) - z * s,
                x * z * (1 - c) + y * s,
            ],
            [
                y * x * (1 - c) + z * s,
                c + y * y * (1 - c),
                y * z * (1 - c) - x * s,
            ],
            [
                z * x * (1 - c) - y * s,
                z * y * (1 - c) + x * s,
                c + z * z * (1 - c),
            ],
        ]
    )


def reflmat(vec: NDArray[float64]) -> NDArray[float64]:
    x, y, z = vec
    return array(
        [
            [1 - 2 * x * x, -2 * x * y, -2 * x * z],
            [-2 * x * y, 1 - 2 * y * y, -2 * y * z],
            [-2 * x * z, -2 * y * z, 1 - 2 * z * z],
        ]
    )
