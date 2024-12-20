"""Utilities."""

__all__ = [
    "clamp",
    "rational",
    "signvar",
    "circshift",
]

from typing import Sequence, Tuple, List


def clamp(val: float, low: float, high: float) -> float:
    """Clamp a value `val` within the interval between `low` and `high`."""
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


def rational(num: float, tol: float) -> Tuple[int, int]:
    """
    Determine the rational representation (nominator and denominator) of a
    decimal number `num` within a tolerance `tol`.
    """
    negative = num < 0.0
    if negative:
        num = -num
    nom = 0
    denom = 1
    while True:
        diff = num - nom / denom
        if abs(diff) <= tol:
            break
        if diff > 0.0:
            nom += 1
        else:
            denom += 1
    if negative:
        nom = -nom
    return nom, denom


def signvar(
    vec: Sequence[float], parity: int = 0, indep: bool = False
) -> List[List[float]]:
    """
    Generate vectors with all possible sign changes of the components of a
    vector `vec` that satisfy a parity `parity`.  If `parity` is
    positive/negative, only the vectors resulting from even/odd number of sign
    changes are returned.  If `parity` is zero, all vectors are returned.  If
    `indep` is enabled, only linearly independent vectors are returned.
    """
    nonzeros = 0
    for comp in vec:
        if comp != 0:
            nonzeros += 1
    arr = []
    excluded = []
    for n in range(2**nonzeros):
        changes = nonzeros * [False]
        i = 0
        sign = 1
        while n > 0:
            if n % 2 == 1:
                changes[i] = True
                sign = -sign
            i += 1
            n //= 2
        if sign * parity >= 0:
            if indep:
                if changes in excluded:
                    continue
                excluded.append([not change for change in changes])
            elem = []
            i = 0
            for comp in vec:
                if comp != 0:
                    if changes[i]:
                        comp = -comp
                    i += 1
                elem.append(comp)
            arr.append(elem)
    return arr


def circshift(vecs: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Generate all possible vectors by applying circular shifts on the
    components of vectors `vecs`.
    """
    arr = []
    for vec in vecs:
        n = len(vec)
        for i in range(n):
            elem = []
            for ii in range(n, 0, -1):
                elem.append(vec[i - ii])
            arr.append(elem)
    return arr
