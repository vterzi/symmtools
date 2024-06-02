"""Constants."""

__all__ = [
    "INF",
    "PI",
    "TAU",
    "PI_2",
    "EPS",
    "TOL",
    "PHI",
    "ORIGIN",
    "PRIMAX",
    "SECAX",
    "TERNAX",
    "INF_SYMB",
    "REFL_SYMB",
    "LABEL_RE",
    "FLOAT_RE",
]

from math import nan, inf, pi

NAN = nan
INF = inf
PI = pi
TAU = 2.0 * pi
PI_2 = 0.5 * pi
EPS = 7.0 / 3.0 - 4.0 / 3.0 - 1.0
TOL = 2**4 * EPS
PHI = 0.5 * (1.0 + 5.0**0.5)
ORIGIN = (0.0, 0.0, 0.0)
PRIMAX = (0.0, 0.0, 1.0)
SECAX = (1.0, 0.0, 0.0)
TERNAX = (0.0, 1.0, 0.0)
INF_SYMB = "oo"  # "\u221e"
REFL_SYMB = "s"  # "\u03c3"
LABEL_RE = r"(?:\b[A-Za-z_]\w*\b)"
FLOAT_RE = r"(?:[+\-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+\-]?\d+)?)"
