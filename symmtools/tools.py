"""Miscellaneous functions."""

__all__ = [
    "rational",
    "signvar",
    "circshift",
]

from .typehints import (
    Tuple,
    List,
    Bool,
    Int,
    Real,
    RealVector,
    RealVectors,
)


def rational(num: float, tol: float) -> Tuple[int, int]:
    """
    Determine the rational representation (nominator and denominator) of a
    decimal number `num` within a tolerance `tol`.
    """
    negative = num < 0.0
    num = abs(num)
    nom = 0
    denom = 1
    diff = num - nom / denom
    while abs(diff) > tol:
        if diff > 0.0:
            nom += 1
        else:
            denom += 1
        diff = num - nom / denom
    if negative:
        nom = -nom
    return nom, denom


def signvar(
    vec: RealVector, parity: Int = 0, indep: Bool = False
) -> List[List[Real]]:
    """
    Generate vectors with all possible sign changes of the coordinates of a
    vector `vec` that satisfy a parity `parity`.  If `parity` is
    positive/negative, only the vectors resulting from even/odd number of sign
    changes are returned.  If `parity` is zero, all vectors are returned.  If
    `indep` is `True`, only linearly independent vectors are returned.
    """
    nonzeros = 0
    for coord in vec:
        if coord != 0:
            nonzeros += 1
    res = []
    excluded = []
    for n in range(2**nonzeros):
        change = nonzeros * [False]
        i = 0
        sign = 1
        while n > 0:
            if n % 2 == 1:
                change[i] = True
                sign = -sign
            i += 1
            n //= 2
        if sign * parity >= 0:
            if indep:
                if change in excluded:
                    continue
                excluded.append([not ch for ch in change])
            new = []
            i = 0
            for coord in vec:
                if coord != 0:
                    if change[i]:
                        coord = -coord
                    i += 1
                new.append(coord)
            res.append(new)
    return res


def circshift(vecs: RealVectors) -> List[List[Real]]:
    """
    Generate all possible vectors by applying circular shifts on the
    coordinates of vectors `vecs`.
    """
    res = []
    for vec in vecs:
        n = len(vec)
        for i in range(n):
            new = []
            for ii in range(n):
                new.append(vec[(i + ii) % n])
            res.append(new)
    return res
