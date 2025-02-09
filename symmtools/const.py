"""Constants."""

__all__ = [
    "NAN",
    "INF",
    "PI",
    "TAU",
    "HALF_PI",
    "EPS",
    "TOL",
    "PHI",
    "ORIGIN",
    "PRIMAX",
    "SECAX",
    "TERTAX",
    "SPECIAL_ANGLES",
    "SPECIAL_COMPONENTS",
    "ORB_TYPES",
    "LABEL_RE",
    "UINT_RE",
    "FLOAT_RE",
    "Symb",
]

from math import (
    nan as NAN,
    inf as INF,
    pi as PI,
    tau as TAU,
    ldexp,
    sqrt,
    atan,
)

HALF_PI = 0.5 * PI
EPS = 7.0 / 3.0 - 4.0 / 3.0 - 1.0
TOL = ldexp(EPS, 10)
PHI = 0.5 * (1.0 + sqrt(5.0))
ORIGIN = (0.0, 0.0, 0.0)
PRIMAX = (0.0, 0.0, 1.0)
SECAX = (1.0, 0.0, 0.0)
TERTAX = (0.0, 1.0, 0.0)
SPECIAL_ANGLES = (
    0.0,
    atan(1.0 / PHI**2),
    atan(1.0 / PHI),
    atan(1.0 / sqrt(2.0)),
    PI / 5.0,
    atan(2.0 / PHI**2),
    atan(2.0 / sqrt(5.0)),
    PI / 4.0,
    atan(sqrt(2.0)),
    atan(PHI),
    PI / 3.0,
    atan(2.0),
    atan(PHI**2),
    atan(2.0 * sqrt(2.0)),
    2.0 * PI / 5.0,
    atan(2.0 * PHI**2),
    PI / 2.0,
)
SPECIAL_COMPONENTS = (
    0.0,
    sqrt(1.0 / 24.0) / PHI**3,
    sqrt(sqrt(1.0 / 180.0) / PHI**5),
    sqrt(1.0 / 18.0) / PHI**2,
    sqrt(1.0 / 12.0) / PHI**2,
    sqrt(1.0 / 24.0) / PHI,
    sqrt(1.0 / 9.0) / PHI**2,
    sqrt(sqrt(1.0 / 180.0) / PHI**3),
    sqrt(1.0 / 8.0) / PHI**2,
    sqrt(sqrt(1.0 / 80.0) / PHI**3),
    sqrt(1.0 / 12.0) / PHI,
    sqrt(sqrt(1.0 / 45.0) / PHI**3),
    sqrt(1.0 / 24.0),
    sqrt(1.0 / 8.0) / PHI,
    sqrt(sqrt(1.0 / 20.0) / PHI**3),
    sqrt(1.0 / 18.0),
    sqrt(1.0 / 6.0) / PHI,
    sqrt(sqrt(1.0 / 80.0) / PHI),
    sqrt(1.0 / 5.0) / PHI,
    sqrt(5.0 / 24.0) / PHI,
    sqrt(1.0 / 12.0),
    sqrt(sqrt(1.0 / 45.0) / PHI),
    sqrt(1.0 / 4.0) / PHI,
    sqrt(1.0 / 24.0) * PHI,
    sqrt(1.0 / 9.0),
    sqrt(1.0 / 8.0),
    sqrt(1.0 / 3.0) / PHI,
    sqrt(sqrt(1.0 / 20.0) / PHI),
    sqrt(3.0 / 8.0) / PHI,
    sqrt(1.0 / 6.0),
    sqrt(sqrt(1.0 / 80.0) * PHI),
    sqrt(1.0 / 5.0),
    sqrt(1.0 / 12.0) * PHI,
    sqrt(2.0 / 9.0),
    sqrt(sqrt(5.0 / 36.0) / PHI),
    sqrt(sqrt(1.0 / 45.0) * PHI),
    sqrt(1.0 / 4.0),
    sqrt(sqrt(1.0 / 5.0) / PHI),
    sqrt(5.0 / 18.0),
    sqrt(sqrt(1.0 / 180.0) * PHI**3),
    sqrt(1.0 / 8.0) * PHI,
    sqrt(1.0 / 3.0),
    sqrt((5.0 / 3.0 - 1.0 / 2.0 / PHI) / 4.0),
    sqrt(sqrt(5.0 / 16.0) / PHI),
    sqrt(sqrt(1.0 / 20.0) * PHI),
    sqrt(sqrt(16.0 / 45.0) / PHI),
    sqrt(3.0 / 8.0),
    sqrt(1.0 / 18.0) * PHI**2,
    sqrt(5.0 / 12.0),
    sqrt(1.0 / 6.0) * PHI,
    sqrt(4.0 / 9.0),
    sqrt((5.0 / 3.0 - 1.0 / 2.0 / PHI) / 3.0),
    sqrt(sqrt(1.0 / 80.0) * PHI**3),
    sqrt(1.0 / 2.0),
    sqrt(1.0 / 5.0) * PHI,
    sqrt(5.0 / 24.0) * PHI,
    sqrt(5.0 / 9.0),
    sqrt(1.0 / 12.0) * PHI**2,
    sqrt(sqrt(5.0 / 36.0) * PHI),
    sqrt((5.0 / 3.0 + 1.0 / 2.0 * PHI) / 4.0),
    sqrt(5.0 / 8.0),
    sqrt(sqrt(1.0 / 45.0) * PHI**3),
    sqrt(1.0 / 4.0) * PHI,
    sqrt(2.0 / 3.0),
    sqrt(sqrt(1.0 / 5.0) * PHI),
    sqrt(1.0 / 24.0) * PHI**3,
    sqrt(3.0 / 4.0),
    sqrt(1.0 / 9.0) * PHI**2,
    sqrt(4.0 / 5.0),
    sqrt((5.0 / 3.0 + 1.0 / 2.0 * PHI) / 3.0),
    sqrt(sqrt(1.0 / 180.0) * PHI**5),
    sqrt(5.0 / 6.0),
    sqrt(1.0 / 8.0) * PHI**2,
    sqrt(1.0 / 3.0) * PHI,
    sqrt(8.0 / 9.0),
    sqrt(sqrt(5.0 / 16.0) * PHI),
    sqrt(sqrt(1.0 / 20.0) * PHI**3),
    sqrt(sqrt(16.0 / 45.0) * PHI),
    sqrt(3.0 / 8.0) * PHI,
    1.0,
)
ORB_TYPES = "spdfghiklmnoqrtuvwxyz"
LABEL_RE = r"(?:\b[A-Za-z_]\w*\b)"
UINT_RE = r"(?:\d+)"
FLOAT_RE = r"(?:[+\-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+\-]?\d+)?)"


class _Symbols:
    _unicode = False

    def use_ascii(self):
        """Use ASCII symbols."""
        self._unicode = False

    def use_unicode(self):
        """Use Unicode symbols."""
        self._unicode = True

    @property
    def INF(self):
        """Infinity symbol."""
        return "\u221e" if self._unicode else "oo"

    @property
    def IDENT(self):
        """Identity element symbol."""
        return "E"

    @property
    def ROT(self):
        """Rotation axis symbol."""
        return "C"

    @property
    def REFL(self):
        """Reflection plane symbol."""
        return "\u03c3" if self._unicode else "s"

    @property
    def INV(self):
        """Inversion center symbol."""
        return "i"

    @property
    def ROTOREFL(self):
        """Rotoreflection axis symbol."""
        return "S"

    @property
    def PT_GRP_ROTS(self):
        """Point group rotation symbols."""
        return "CSDTOIK"

    @property
    def PT_GRP_REFLS(self):
        """Point group reflection symbols."""
        return " sivdh"


Symb = _Symbols()
