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
    "INF_SYMB",
    "REFL_SYMB",
]

from math import nan, inf, pi

NAN = nan
INF = inf
PI = pi
TAU = 2.0 * pi
EPS = 7.0 / 3.0 - 4.0 / 3.0 - 1.0
TOL = 2**4 * EPS
PHI = 0.5 * (1.0 + 5.0**0.5)
ORIGIN = (0.0, 0.0, 0.0)
PRIMAX = (0.0, 0.0, 1.0)
SECAX = (1.0, 0.0, 0.0)
TERNAX = (0.0, 1.0, 0.0)
INF_SYMB = "oo"  # "\u221e"
REFL_SYMB = "s"  # "\u03c3"
