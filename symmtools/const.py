"""Constants."""

__all__ = [
    "EPS",
    "TOL",
    "PHI",
    "ORIGIN",
    "PRIMAX",
    "SECAX",
    "TERNAX",
]

EPS = 7 / 3 - 4 / 3 - 1
TOL = 2**4 * EPS
PHI = (1 + 5**0.5) / 2
ORIGIN = (0, 0, 0)
PRIMAX = (0, 0, 1)
SECAX = (1, 0, 0)
TERNAX = (0, 1, 0)
