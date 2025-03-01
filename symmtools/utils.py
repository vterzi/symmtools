"""Utilities."""

__all__ = [
    "clamp",
    "rational",
    "reducefrac",
    "sqrtfactor",
    "signvar",
    "circshift",
    "linassign",
]

from math import gcd
from typing import Sequence, Tuple, List

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore

    LINEAR_SUM_ASSIGNMENT_AVAIL = True
except ImportError:
    LINEAR_SUM_ASSIGNMENT_AVAIL = False

from .const import INF


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


def reducefrac(nom: int, denom: int) -> Tuple[int, int]:
    """
    Reduce a fraction given by a nominator `nom` and a denominator `denom`.
    """
    divisor = gcd(nom, denom)
    if divisor > 1:
        nom //= divisor
        denom //= divisor
    return nom, denom


def sqrtfactor(radicand: int) -> Tuple[int, int]:
    """Factor out the square factors of a square-root radicand `radicand`."""
    prefactor = 1
    factor = 1
    while True:
        factor += 1
        sq_factor = factor * factor
        if sq_factor > radicand:
            break
        while radicand % sq_factor == 0:
            prefactor *= factor
            radicand //= sq_factor
    return prefactor, radicand


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


def linassign(
    mat: Sequence[Sequence[float]],
    maximize: bool = False,
    fast: bool = True,
) -> Tuple[int, ...]:
    """
    Solve the linear balanced assignment problem for a square matrix `mat` and
    return the column indices of the optimal assignment for each row.  If
    `maximize` is enabled, the maximum assignment is sought instead of the
    minimum.  If `fast` is enabled, a faster algorithm is used.
    """
    if LINEAR_SUM_ASSIGNMENT_AVAIL and fast:
        return tuple(linear_sum_assignment(mat, maximize)[1])
    arr = list(map(list, mat))
    n = len(arr)
    if maximize:
        for row in arr:
            if len(row) != n:
                raise ValueError("non-square matrix")
            max_val = max(row)
            if max_val == INF:
                raise ValueError("infinite value")
            for i_col in range(n):
                row[i_col] = max_val - row[i_col]
    else:
        for row in arr:
            if len(row) != n:
                raise ValueError("non-square matrix")
            min_val = min(row)
            if min_val == -INF:
                raise ValueError("infinite value")
            for i_col in range(n):
                row[i_col] -= min_val
    for i_col in range(n):
        min_val = min(row[i_col] for row in arr)
        if min_val > 0.0:
            for row in arr:
                row[i_col] -= min_val
    assignment = [-1] * n
    col_idxs = list(range(n))
    for i_row, row in enumerate(arr):
        for i_col in col_idxs:
            if row[i_col] == 0.0:
                assignment[i_row] = i_col
                col_idxs.remove(i_col)
                break
    while True:
        candidates = [-1] * n
        covered_rows = [False] * n
        covered_cols = [False] * n
        for i_col in assignment:
            if i_col != -1:
                covered_cols[i_col] = True
        if all(covered_cols):
            break
        repeat = True
        while repeat:
            min_val = INF
            for i_row, covered_row in enumerate(covered_rows):
                if not covered_row:
                    row = arr[i_row]
                    for i_col, covered_col in enumerate(covered_cols):
                        if not covered_col:
                            val = row[i_col]
                            if val < min_val:
                                min_val = val
                                if val == 0.0:
                                    candidates[i_row] = i_col
                                    if assignment[i_row] != -1:
                                        covered_rows[i_row] = True
                                        covered_cols[assignment[i_row]] = False
                                    else:
                                        while i_row != -1:
                                            new_i_row = (
                                                assignment.index(i_col)
                                                if i_col in assignment
                                                else -1
                                            )
                                            assignment[i_row] = i_col
                                            i_row = new_i_row
                                            i_col = candidates[i_row]
                                        repeat = False
                                    break
                    else:
                        continue
                    break
            else:
                for i_row, covered_row in enumerate(covered_rows):
                    row = arr[i_row]
                    for i_col, covered_col in enumerate(covered_cols):
                        if not covered_row and not covered_col:
                            row[i_col] -= min_val
                        elif covered_row and covered_col:
                            row[i_col] += min_val
    return tuple(assignment)
