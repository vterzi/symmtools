"""Class for point groups."""

__all__ = ["PointGroupInfo", "PointGroupInfos", "VARIANTS"]

from abc import ABC, abstractmethod
from math import sin, cos
from typing import (
    Sequence,
    Iterable,
    Iterator,
    FrozenSet,
    Tuple,
    List,
    Dict,
)

from .const import (
    INF,
    PI,
    PI_2,
    PHI,
    PRIMAX,
    SECAX,
    SPECIAL_ANGLES,
    SPECIAL_COMPONENTS,
)
from .linalg3d import Vector, vector, lincomb3, intersectangle
from .utils import signvar, circshift
from .symmelem import (
    SymmetryElement,
    InversionCenter,
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    CenterRotationAxes,
    AxisReflectionPlanes,
    CenterReflectionPlanes,
    CenterRotoreflectionAxes,
    VectorSymmetryElement,
    VEC_SYMM_ELEMS,
    labeled_symm_elem,
)

_T_ROT3_VECS = signvar((1.0, 1.0, 1.0), 1)
_T_ROT2_VECS = circshift(((1.0, 0.0, 0.0),))
_TD_REFL_VECS = circshift(signvar((1.0, 1.0, 0.0), 0, True))
_TD_ROTOREFL4_VECS = _T_ROT2_VECS
_TH_REFL_VECS = _T_ROT2_VECS
_TH_ROTOREFL6_VECS = _T_ROT3_VECS
_O_ROT4_VECS = _T_ROT2_VECS
_O_ROT3_VECS = _T_ROT3_VECS
_O_ROT2_VECS = _TD_REFL_VECS
_OH_REFL_H_VECS = _O_ROT4_VECS
_OH_REFL_D_VECS = _O_ROT2_VECS
_OH_ROTOREFL6_VECS = _O_ROT3_VECS
_OH_ROTOREFL4_VECS = _O_ROT4_VECS
_I_ROT5_VECS = circshift(signvar((PHI, 1.0, 0.0), 0, True))
_I_ROT3_VECS = _T_ROT3_VECS + circshift(
    signvar((1.0, 1.0 + PHI, 0.0), 0, True)
)
_I_ROT2_VECS = _T_ROT2_VECS + circshift(
    signvar((1.0, PHI, 1.0 + PHI), 0, True)
)
_IH_REFL_VECS = _I_ROT2_VECS
_IH_ROTOREFL10_VECS = _I_ROT5_VECS
_IH_ROTOREFL6_VECS = _I_ROT3_VECS


_Indices = Sequence[Tuple[int, int, int]]


class _Vecs(ABC):
    symb = ""

    def __init__(
        self, idxs1: _Indices, idxs2: _Indices, idxs3: _Indices
    ) -> None:
        self.vecs = tuple(
            tuple(
                tuple(
                    (
                        SPECIAL_COMPONENTS[idx]
                        if idx >= 0
                        else -SPECIAL_COMPONENTS[-idx]
                    )
                    for idx in idxs
                )
                for idxs in arr_idxs
            )
            for arr_idxs in (idxs1, idxs2, idxs3)
        )

    @abstractmethod
    def symm_elems(
        self, ax1: Vector, ax2: Vector, ax3: Vector, suffix: str
    ) -> Iterator[SymmetryElement]:
        pass


class _VecsT(_Vecs):
    symb = "T"

    def symm_elems(
        self, ax1: Vector, ax2: Vector, ax3: Vector, suffix: str
    ) -> Iterator[SymmetryElement]:
        diag = suffix == "d"
        horiz = suffix == "h"
        for vec in self.vecs[0]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 3)
            if horiz:
                yield RotoreflectionAxis(vec, 6)
        for vec in self.vecs[1]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 2)
            if diag:
                yield RotoreflectionAxis(vec, 4)
            elif horiz:
                yield ReflectionPlane(vec)
        if diag:
            for vec in self.vecs[2]:
                vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
                yield ReflectionPlane(vec)


class _VecsO(_Vecs):
    symb = "O"

    def symm_elems(
        self, ax1: Vector, ax2: Vector, ax3: Vector, suffix: str
    ) -> Iterator[SymmetryElement]:
        horiz = suffix == "h"
        for vec in self.vecs[0]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 4)
            if horiz:
                yield RotoreflectionAxis(vec, 4)
                yield ReflectionPlane(vec)
        for vec in self.vecs[1]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 3)
            if horiz:
                yield RotoreflectionAxis(vec, 6)
        for vec in self.vecs[2]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 2)
            if horiz:
                yield ReflectionPlane(vec)


class _VecsI(_Vecs):
    symb = "I"

    def symm_elems(
        self, ax1: Vector, ax2: Vector, ax3: Vector, suffix: str
    ) -> Iterator[SymmetryElement]:
        horiz = suffix == "h"
        for vec in self.vecs[0]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 5)
            if horiz:
                yield RotoreflectionAxis(vec, 10)
        for vec in self.vecs[1]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 3)
            if horiz:
                yield RotoreflectionAxis(vec, 6)
        for vec in self.vecs[2]:
            vec = lincomb3(ax1, vec[0], ax2, vec[1], ax3, vec[2])
            yield RotationAxis(vec, 2)
            if horiz:
                yield ReflectionPlane(vec)


_T_ROT3_IDXS_0 = (
    (41, 41, 41),
    (41, 41, -41),
    (41, -41, 41),
    (-41, 41, 41),
)
_T_ROT2_IDXS_0 = (
    (79, 0, 0),
    (0, 79, 0),
    (0, 0, 79),
)
_TD_REFL_IDXS_0 = (
    (53, 53, 0),
    (53, 0, 53),
    (0, 53, 53),
    (53, -53, 0),
    (53, 0, -53),
    (0, 53, -53),
)
_T_ROT3_IDXS_1 = (
    (79, 0, 0),
    (24, 74, 0),
    (24, -33, 63),
    (-24, 33, 63),
)
_T_ROT2_IDXS_1 = (
    (41, -63, 0),
    (41, 29, 53),
    (41, 29, -53),
)
_TD_REFL_IDXS_1 = (
    (63, 41, 0),
    (63, -20, 36),
    (-63, 20, 36),
    (0, 66, 36),
    (0, 66, -36),
    (0, 0, 79),
)
_T_VECS = (
    _VecsT(_T_ROT3_IDXS_0, _T_ROT2_IDXS_0, _TD_REFL_IDXS_0),
    _VecsT(_T_ROT3_IDXS_1, _T_ROT2_IDXS_1, _TD_REFL_IDXS_1),
)
_O_VECS = (
    _VecsO(_T_ROT2_IDXS_0, _T_ROT3_IDXS_0, _TD_REFL_IDXS_0),
    _VecsO(_T_ROT2_IDXS_1, _T_ROT3_IDXS_1, _TD_REFL_IDXS_1),
    _VecsO(
        (
            (79, 0, 0),
            (0, 53, 53),
            (0, 53, -53),
        ),
        (
            (41, 63, 0),
            (41, 0, 63),
            (41, -63, 0),
            (41, 0, -63),
        ),
        (
            (0, 79, 0),
            (0, 0, 79),
            (53, 36, 36),
            (53, 36, -36),
            (53, -36, 36),
            (-53, 36, 36),
        ),
    ),
)
_I_VECS = (
    _VecsI(
        (
            (64, 37, 0),
            (37, 0, 64),
            (0, 64, 37),
            (64, -37, 0),
            (-37, 0, 64),
            (0, 64, -37),
        ),
        _T_ROT3_IDXS_0
        + (
            (26, 73, 0),
            (73, 0, 26),
            (0, 26, 73),
            (26, -73, 0),
            (-73, 0, 26),
            (0, 26, -73),
        ),
        _T_ROT2_IDXS_0
        + (
            (62, 22, 36),
            (62, 22, -36),
            (62, -22, 36),
            (-62, 22, 36),
            (36, 62, 22),
            (36, 62, -22),
            (36, -62, 22),
            (-36, 62, 22),
            (22, 36, 62),
            (22, 36, -62),
            (22, -36, 62),
            (-22, 36, 62),
        ),
    ),
    _VecsI(
        (
            (61, 39, -14),
            (61, -2, 44),
            (-61, 34, 27),
            (11, 58, 44),
            (11, 7, -76),
            (11, -70, 27),
        ),
        _T_ROT3_IDXS_1
        + (
            (56, 38, 29),
            (56, 3, -49),
            (56, -47, 16),
            (24, 15, 71),
            (24, 51, -49),
            (-24, 69, 16),
        ),
        _T_ROT2_IDXS_1
        + (
            (73, 19, 13),
            (73, 1, -25),
            (73, -23, 8),
            (41, 59, 13),
            (41, -42, 40),
            (-41, 12, 60),
            (26, 65, -25),
            (26, -5, 72),
            (-26, 55, 40),
            (0, 78, 8),
            (0, 28, 72),
            (0, 46, -60),
        ),
    ),
    _VecsI(
        (
            (79, 0, 0),
            (31, 68, 0),
            (31, 18, 64),
            (31, 18, -64),
            (31, -54, 37),
            (-31, 54, 37),
        ),
        (
            (61, -45, 0),
            (61, 35, 26),
            (61, 35, -26),
            (61, -11, 41),
            (-61, 11, 41),
            (11, -77, 0),
            (11, 61, 41),
            (11, 61, -41),
            (11, -21, 73),
            (-11, 21, 73),
        ),
        (
            (64, 37, 0),
            (64, 9, 36),
            (64, 9, -36),
            (64, -30, 22),
            (-64, 30, 22),
            (37, -64, 0),
            (37, 52, 36),
            (37, 52, -36),
            (37, -17, 62),
            (-37, 17, 62),
            (0, 43, 62),
            (0, 43, -62),
            (0, 75, 22),
            (0, -75, 22),
            (0, 0, 79),
        ),
    ),
    _VecsI(
        (
            (61, 21, 37),
            (61, 21, -37),
            (61, -45, 0),
            (11, 77, 0),
            (11, -35, 64),
            (-11, 35, 64),
        ),
        (
            (79, 0, 0),
            (56, 50, 0),
            (56, -24, 41),
            (-56, 24, 41),
            (24, 56, 41),
            (24, 56, -41),
            (24, -67, 26),
            (-24, 67, 26),
            (24, 6, -73),
            (24, 6, 73),
        ),
        (
            (73, 26, 0),
            (73, -10, 22),
            (-73, 10, 22),
            (41, 57, 22),
            (41, 57, -22),
            (41, -48, 36),
            (-41, 48, 36),
            (41, -4, 62),
            (-41, 4, 62),
            (26, -73, 0),
            (26, 32, 62),
            (26, 32, -62),
            (0, 66, 36),
            (0, 66, -36),
            (0, 0, 79),
        ),
    ),
)
VARIANTS: Dict[
    Tuple[int, int],
    Dict[float, Tuple[Tuple[_Vecs, Tuple[int, int, int]], ...]],
] = {
    (5, 5): {SPECIAL_ANGLES[11]: ((_I_VECS[2], (0, 1, 2)),)},
    (5, 3): {
        SPECIAL_ANGLES[5]: ((_I_VECS[2], (0, -1, 2)),),
        SPECIAL_ANGLES[15]: ((_I_VECS[2], (0, -1, 2)),),
    },
    (5, 2): {
        SPECIAL_ANGLES[2]: ((_I_VECS[2], (0, 1, 2)),),
        SPECIAL_ANGLES[9]: ((_I_VECS[2], (0, -1, 2)),),
        SPECIAL_ANGLES[16]: ((_I_VECS[1], (0, 2, 1)),),
    },
    (4, 4): {SPECIAL_ANGLES[16]: ((_O_VECS[0], (0, 1, 2)),)},
    (4, 3): {SPECIAL_ANGLES[8]: ((_O_VECS[2], (0, 1, 2)),)},
    (4, 2): {
        SPECIAL_ANGLES[7]: ((_O_VECS[0], (0, 1, 2)),),
        SPECIAL_ANGLES[16]: ((_O_VECS[2], (0, 1, 2)),),
    },
    (3, 3): {
        SPECIAL_ANGLES[6]: ((_I_VECS[3], (0, 1, 2)),),
        SPECIAL_ANGLES[13]: (
            (_I_VECS[1], (0, 1, 2)),
            (_O_VECS[1], (0, 1, 2)),
            (_T_VECS[1], (0, 1, 2)),
        ),
    },
    (3, 2): {
        SPECIAL_ANGLES[1]: ((_I_VECS[3], (0, 1, 2)),),
        SPECIAL_ANGLES[3]: ((_O_VECS[1], (0, 1, 2)),),
        SPECIAL_ANGLES[8]: (
            (_I_VECS[1], (0, -1, 2)),
            (_T_VECS[1], (0, -1, 2)),
        ),
        SPECIAL_ANGLES[12]: ((_I_VECS[3], (0, -1, 2)),),
        SPECIAL_ANGLES[16]: (
            (_I_VECS[3], (0, 2, -1)),
            (_O_VECS[1], (0, 2, -1)),
        ),
    },
    (2, 2): {
        SPECIAL_ANGLES[4]: ((_I_VECS[2], (2, 1, 0)),),
        SPECIAL_ANGLES[10]: (
            (_I_VECS[3], (2, 1, 0)),
            (_O_VECS[0], (0, 1, 2)),
        ),
        SPECIAL_ANGLES[14]: ((_I_VECS[2], (2, -1, 0)),),
        SPECIAL_ANGLES[16]: (
            (_I_VECS[0], (0, 1, 2)),
            (_O_VECS[2], (2, 0, 1)),
            (_T_VECS[0], (0, 1, 2)),
        ),
    },
}


class PointGroupInfo:
    _symm_elems: Tuple[SymmetryElement, ...] = ()
    _types: Dict[Tuple, int] = {}
    _angles: Dict[FrozenSet[Tuple], Dict[float, int]] = {}

    @property
    def symm_elems(self) -> Iterable[SymmetryElement]:
        return self._symm_elems

    @property
    def types(self) -> Dict[Tuple, int]:
        return self._types.copy()

    @property
    def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
        return {props: angles.copy() for props, angles in self._angles.items()}


class SpecialPointGroupInfo(PointGroupInfo):
    def __init__(self, symm_elems: Iterable[SymmetryElement]) -> None:
        self._symm_elems = tuple(symm_elems)
        vec_symm_elems: List[VectorSymmetryElement] = []
        types: Dict[Tuple, int] = {}
        angles: Dict[FrozenSet[Tuple], Dict[float, int]] = {}
        for symm_elem in self._symm_elems:
            prop = symm_elem.props
            if prop not in types:
                types[prop] = 0
            types[prop] += 1
            if isinstance(symm_elem, VEC_SYMM_ELEMS):
                vec = symm_elem.vec
                for vec_symm_elem in vec_symm_elems:
                    ang = intersectangle(vec, vec_symm_elem.vec)
                    min_diff = INF
                    best_ang = 0.0
                    for special_ang in SPECIAL_ANGLES:
                        diff = abs(ang - special_ang)
                        if diff < min_diff:
                            min_diff = diff
                            best_ang = special_ang
                    ang = best_ang
                    props = frozenset((prop, vec_symm_elem.props))
                    if props not in angles:
                        angles[props] = {}
                    if ang not in angles[props]:
                        angles[props][ang] = 0
                    angles[props][ang] += 1
                vec_symm_elems.append(symm_elem)
        self._types = types
        self._angles = angles


class PointGroupInfos:
    C1 = SpecialPointGroupInfo(())
    Cs = SpecialPointGroupInfo((ReflectionPlane(PRIMAX),))
    Ci = SpecialPointGroupInfo((InversionCenter(),))

    class Cn(PointGroupInfo):
        def __init__(self, order: int) -> None:
            if order < 2:
                raise ValueError("order less than 2")
            self._order = order

        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield RotationAxis(PRIMAX, self._order)

        @property
        def types(self) -> Dict[Tuple, int]:
            return {(RotationAxis, self._order): 1}

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            return {}

    class Cnv(Cn):
        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield from super().symm_elems
            order = self._order
            step = PI / order
            yield labeled_symm_elem(ReflectionPlane(SECAX), "v")
            if order % 2 == 1:
                ang = step
                for _ in range(1, order):
                    yield labeled_symm_elem(
                        ReflectionPlane((cos(ang), sin(ang), 0.0)), "v"
                    )
                    ang += step
            else:
                ang = step
                step += step
                for _ in range(1, order, 2):
                    yield labeled_symm_elem(
                        ReflectionPlane((cos(ang), sin(ang), 0.0)), "d"
                    )
                    ang += step
                ang = step
                for _ in range(2, order, 2):
                    yield labeled_symm_elem(
                        ReflectionPlane((cos(ang), sin(ang), 0.0)), "v"
                    )
                    ang += step

        @property
        def types(self) -> Dict[Tuple, int]:
            res = super().types
            res[(ReflectionPlane,)] = self._order
            return res

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            res = super().angles
            order = self._order
            sv = (ReflectionPlane,)
            res[frozenset(((RotationAxis, order), sv))] = {PI_2: order}
            entry = {}
            step = PI / order
            ang = step
            for _ in range(1, (order + 1) // 2):
                entry[ang] = order
                ang += step
            if order % 2 == 0:
                entry[PI_2] = order // 2
            res[frozenset((sv,))] = entry
            return res

    class Cnh(Cn):
        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield from super().symm_elems
            order = self._order
            yield labeled_symm_elem(ReflectionPlane(PRIMAX), "h")
            if order % 2 == 0:
                yield InversionCenter()
            if order > 2:
                yield RotoreflectionAxis(PRIMAX, order)

        @property
        def types(self) -> Dict[Tuple, int]:
            res = super().types
            order = self._order
            res[(ReflectionPlane,)] = 1
            if order % 2 == 0:
                res[(InversionCenter,)] = 1
            if order > 2:
                res[(RotoreflectionAxis, order)] = 1
            return res

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            res = super().angles
            order = self._order
            Cn = (RotationAxis, order)
            sh = (ReflectionPlane,)
            res[frozenset((Cn, sh))] = {0.0: 1}
            if order > 2:
                Sn = (RotoreflectionAxis, order)
                res[frozenset((Cn, Sn))] = {0.0: 1}
                res[frozenset((sh, Sn))] = {0.0: 1}
            return res

    class S2n(Cn):
        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield from super().symm_elems
            order = self._order
            if order % 2 == 1:
                yield InversionCenter()
            yield RotoreflectionAxis(PRIMAX, 2 * order)

        @property
        def types(self) -> Dict[Tuple, int]:
            res = super().types
            order = self._order
            if order % 2 == 1:
                res[(InversionCenter,)] = 1
            res[(RotoreflectionAxis, 2 * order)] = 1
            return res

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            res = super().angles
            order = self._order
            res[
                frozenset(
                    ((RotationAxis, order), (RotoreflectionAxis, 2 * order))
                )
            ] = {0.0: 1}
            return res

    class Dn(Cn):
        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield from super().symm_elems
            order = self._order
            step = PI / order
            yield labeled_symm_elem(RotationAxis(SECAX, 2), "'")
            if order % 2 == 1:
                ang = step
                for _ in range(1, order):
                    yield labeled_symm_elem(
                        RotationAxis((cos(ang), sin(ang), 0.0), 2), "'"
                    )
                    ang += step
            else:
                ang = step
                step += step
                for _ in range(1, order, 2):
                    yield labeled_symm_elem(
                        RotationAxis((cos(ang), sin(ang), 0.0), 2), "''"
                    )
                    ang += step
                ang = step
                for _ in range(2, order, 2):
                    yield labeled_symm_elem(
                        RotationAxis((cos(ang), sin(ang), 0.0), 2), "'"
                    )
                    ang += step

        @property
        def types(self) -> Dict[Tuple, int]:
            res = super().types
            order = self._order
            res[(RotationAxis, 2)] = order + (0 if order > 2 else 1)
            return res

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            res = super().angles
            order = self._order
            C2 = (RotationAxis, 2)
            if order > 2:
                res[frozenset(((RotationAxis, order), C2))] = {PI_2: order}
                entry = {}
                step = PI / order
                ang = step
                for _ in range(1, (order + 1) // 2):
                    entry[ang] = order
                    ang += step
                if order % 2 == 0:
                    entry[PI_2] = order // 2
                res[frozenset((C2,))] = entry
            else:
                res[frozenset((C2,))] = {PI_2: 3}
            return res

    class Dnd(Dn):
        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield from super().symm_elems
            order = self._order
            step = PI / order
            ang = 0.5 * step
            for _ in range(order):
                yield labeled_symm_elem(
                    ReflectionPlane((cos(ang), sin(ang), 0.0)), "d"
                )
                ang += step
            if order % 2 == 1:
                yield InversionCenter()
            yield RotoreflectionAxis(PRIMAX, 2 * order)

        @property
        def types(self) -> Dict[Tuple, int]:
            res = super().types
            order = self._order
            res[(ReflectionPlane,)] = order
            if order % 2 == 1:
                res[(InversionCenter,)] = 1
            res[(RotoreflectionAxis, 2 * order)] = 1
            return res

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            res = super().angles
            order = self._order
            double_order = 2 * order
            C2 = (RotationAxis, 2)
            sd = (ReflectionPlane,)
            S2n = (RotoreflectionAxis, double_order)
            if order > 2:
                Cn = (RotationAxis, order)
                res[frozenset((Cn, sd))] = {PI_2: order}
                entry = {}
                step = PI / double_order
                ang = step
                for _ in range(1, order, 2):
                    entry[ang] = double_order
                    ang += step
                if order % 2 == 1:
                    entry[PI_2] = order
                res[frozenset((C2, sd))] = entry
                res[frozenset((Cn, S2n))] = {0.0: 1}
                res[frozenset((C2, S2n))] = {PI_2: order}
            else:
                res[frozenset((C2, sd))] = {PI_2: 2, 0.5 * PI_2: 4}
                res[frozenset((C2, S2n))] = {0.0: 1, PI_2: 2}
            entry = {}
            step = PI / order
            ang = step
            for _ in range(1, (order + 1) // 2):
                entry[ang] = order
                ang += step
            if order % 2 == 0:
                entry[PI_2] = order // 2
            res[frozenset((sd,))] = entry
            res[frozenset((sd, S2n))] = {PI_2: order}
            return res

    class Dnh(Dn):
        @property
        def symm_elems(self) -> Iterable[SymmetryElement]:
            yield from super().symm_elems
            order = self._order
            yield labeled_symm_elem(ReflectionPlane(PRIMAX), "h")
            step = PI / order
            yield labeled_symm_elem(ReflectionPlane(SECAX), "v")
            if order % 2 == 1:
                ang = step
                for _ in range(1, order):
                    yield labeled_symm_elem(
                        ReflectionPlane((cos(ang), sin(ang), 0.0)), "v"
                    )
                    ang += step
            else:
                ang = step
                step += step
                for _ in range(1, order, 2):
                    yield labeled_symm_elem(
                        ReflectionPlane((cos(ang), sin(ang), 0.0)), "d"
                    )
                    ang += step
                ang = step
                for _ in range(2, order, 2):
                    yield labeled_symm_elem(
                        ReflectionPlane((cos(ang), sin(ang), 0.0)), "v"
                    )
                    ang += step
                yield InversionCenter()
            if order > 2:
                yield RotoreflectionAxis(PRIMAX, order)

        @property
        def types(self) -> Dict[Tuple, int]:
            res = super().types
            order = self._order
            res[(ReflectionPlane,)] = order + 1
            if order % 2 == 0:
                res[(InversionCenter,)] = 1
            if order > 2:
                res[(RotoreflectionAxis, order)] = 1
            return res

        @property
        def angles(self) -> Dict[FrozenSet[Tuple], Dict[float, int]]:
            res = super().angles
            order = self._order
            C2 = (RotationAxis, 2)
            s = (ReflectionPlane,)
            if order > 2:
                double_order = 2 * order
                Cn = (RotationAxis, order)
                Sn = (RotoreflectionAxis, order)
                res[frozenset((Cn, s))] = {0.0: 1, PI_2: order}
                entry1 = {0.0: order}
                entry2 = {}
                step = PI / order
                ang = step
                for _ in range(1, (order + 1) // 2):
                    entry1[ang] = double_order
                    entry2[ang] = order
                    ang += step
                if order % 2 == 0:
                    entry1[PI_2] = double_order
                    entry2[PI_2] = order + order // 2
                else:
                    entry1[PI_2] = order
                    entry2[PI_2] = order
                res[frozenset((C2, s))] = entry1
                res[frozenset((s,))] = entry2
                res[frozenset((Cn, Sn))] = {0.0: 1}
                res[frozenset((C2, Sn))] = {PI_2: order}
                res[frozenset((s, Sn))] = {0.0: 1, PI_2: order}
            else:
                res[frozenset((C2, s))] = {0.0: 3, PI_2: 6}
                res[frozenset((s,))] = {PI_2: 3}
            return res

    T = SpecialPointGroupInfo(
        tuple(RotationAxis(vector(vec), 3) for vec in _T_ROT3_VECS)
        + tuple(RotationAxis(vector(vec), 2) for vec in _T_ROT2_VECS)
    )
    Td = SpecialPointGroupInfo(
        (
            T._symm_elems
            + tuple(
                labeled_symm_elem(ReflectionPlane(vector(vec)), "d")
                for vec in _TD_REFL_VECS
            )
            + tuple(
                RotoreflectionAxis(vector(vec), 4)
                for vec in _TD_ROTOREFL4_VECS
            )
        )
    )
    Th = SpecialPointGroupInfo(
        (
            T._symm_elems
            + tuple(
                labeled_symm_elem(ReflectionPlane(vector(vec)), "h")
                for vec in _TH_REFL_VECS
            )
            + (InversionCenter(),)
            + tuple(
                RotoreflectionAxis(vector(vec), 6)
                for vec in _TH_ROTOREFL6_VECS
            )
        )
    )
    O = SpecialPointGroupInfo(  # noqa: E741
        (
            tuple(RotationAxis(vector(vec), 4) for vec in _O_ROT4_VECS)
            + tuple(RotationAxis(vector(vec), 3) for vec in _O_ROT3_VECS)
            + tuple(
                labeled_symm_elem(RotationAxis(vector(vec), 2), "'")
                for vec in _O_ROT2_VECS
            )
        )
    )
    Oh = SpecialPointGroupInfo(
        (
            O._symm_elems
            + tuple(
                labeled_symm_elem(ReflectionPlane(vector(vec)), "h")
                for vec in _OH_REFL_H_VECS
            )
            + tuple(
                labeled_symm_elem(ReflectionPlane(vector(vec)), "d")
                for vec in _OH_REFL_D_VECS
            )
            + (InversionCenter(),)
            + tuple(
                RotoreflectionAxis(vector(vec), 6)
                for vec in _OH_ROTOREFL6_VECS
            )
            + tuple(
                RotoreflectionAxis(vector(vec), 4)
                for vec in _OH_ROTOREFL4_VECS
            )
        )
    )
    I = SpecialPointGroupInfo(  # noqa: E741
        (
            tuple(RotationAxis(vector(vec), 5) for vec in _I_ROT5_VECS)
            + tuple(RotationAxis(vector(vec), 3) for vec in _I_ROT3_VECS)
            + tuple(RotationAxis(vector(vec), 2) for vec in _I_ROT2_VECS)
        )
    )
    Ih = SpecialPointGroupInfo(
        (
            I._symm_elems
            + tuple(ReflectionPlane(vector(vec)) for vec in _IH_REFL_VECS)
            + (InversionCenter(),)
            + tuple(
                RotoreflectionAxis(vector(vec), 10)
                for vec in _IH_ROTOREFL10_VECS
            )
            + tuple(
                RotoreflectionAxis(vector(vec), 6)
                for vec in _IH_ROTOREFL6_VECS
            )
        )
    )
    Coo = SpecialPointGroupInfo((InfRotationAxis(PRIMAX),))
    Coov = SpecialPointGroupInfo(
        Coo._symm_elems + (AxisReflectionPlanes(PRIMAX),)
    )
    Cooh = SpecialPointGroupInfo(
        Coo._symm_elems
        + (
            ReflectionPlane(PRIMAX),
            InversionCenter(),
            InfRotoreflectionAxis(PRIMAX),
        )
    )
    Doo = SpecialPointGroupInfo(
        (
            InfRotationAxis(PRIMAX),
            AxisRotationAxes(PRIMAX),
        )
    )
    Dooh = SpecialPointGroupInfo(
        Doo._symm_elems
        + (
            AxisReflectionPlanes(PRIMAX),
            ReflectionPlane(PRIMAX),
            InversionCenter(),
            InfRotoreflectionAxis(PRIMAX),
        )
    )
    K = SpecialPointGroupInfo((CenterRotationAxes(),))
    Kh = SpecialPointGroupInfo(
        K._symm_elems
        + (
            CenterReflectionPlanes(),
            InversionCenter(),
            CenterRotoreflectionAxes(),
        )
    )
