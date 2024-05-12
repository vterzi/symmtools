"""Constants."""

__all__ = [
    "INF",
    "PI",
    "TAU",
    "EPS",
    "TOL",
    "PHI",
    "ORIGIN",
    "PRIMAX",
    "SECAX",
    "TERNAX",
]

from math import inf, pi

INF = inf
PI = pi
TAU = 2 * pi
EPS = 7 / 3 - 4 / 3 - 1
TOL = 2**4 * EPS
PHI = (1 + 5**0.5) / 2
ORIGIN = (0, 0, 0)
PRIMAX = (0, 0, 1)
SECAX = (1, 0, 0)
TERNAX = (0, 1, 0)
