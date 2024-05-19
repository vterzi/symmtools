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
TAU = 2.0 * pi
EPS = 7.0 / 3.0 - 4.0 / 3.0 - 1.0
TOL = 2**4 * EPS
PHI = (1.0 + 5.0**0.5) / 2.0
ORIGIN = (0.0, 0.0, 0.0)
PRIMAX = (0.0, 0.0, 1.0)
SECAX = (1.0, 0.0, 0.0)
TERNAX = (0.0, 1.0, 0.0)
