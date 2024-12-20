"""Miscellaneous tools."""

__all__ = ["Plot"]

from random import choice
from typing import Optional, Sequence, Tuple, List

from matplotlib.pyplot import figure  # type: ignore

from .const import EPS, PRIMAX, SECAX
from .linalg3d import Vector, neg, add, sub, mul, norm, normalize, cross
from .primitive import Point, LabeledPoint

_TOL = 2**6 * EPS
ELEM_SYMBS = (
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
ATOMIC_RADII = (
    0,
    32,
    46,
    133,
    102,
    85,
    75,
    71,
    63,
    64,
    67,
    155,
    139,
    126,
    116,
    111,
    103,
    99,
    96,
    196,
    171,
    148,
    136,
    134,
    122,
    119,
    116,
    111,
    110,
    112,
    118,
    124,
    121,
    121,
    116,
    114,
    117,
    210,
    185,
    163,
    154,
    147,
    138,
    128,
    125,
    125,
    120,
    128,
    136,
    142,
    140,
    140,
    136,
    133,
    131,
    232,
    196,
    180,
    163,
    176,
    174,
    173,
    172,
    168,
    169,
    168,
    167,
    166,
    165,
    164,
    170,
    162,
    152,
    146,
    137,
    131,
    129,
    122,
    123,
    124,
    133,
    144,
    144,
    151,
    145,
    147,
    142,
    223,
    201,
    186,
    175,
    169,
    170,
    171,
    172,
    166,
    166,
    168,
    168,
    165,
    167,
    173,
    176,
    161,
    157,
    149,
    143,
    141,
    134,
    129,
    128,
    121,
    122,
    136,
    143,
    162,
    175,
    165,
    157,
)
ELEM_COLORS = (
    "",
    "#eeeeee",
    "#d9ffff",
    "#cc80ff",
    "#c2ff00",
    "#ffb5b5",
    "#909090",
    "#3050f8",
    "#ff0d0d",
    "#90e050",
    "#b3e3f5",
    "#ab5cf2",
    "#8aff00",
    "#bfa6a6",
    "#f0c8a0",
    "#ff8000",
    "#ffff30",
    "#1ff01f",
    "#80d1e3",
    "#8f40d4",
    "#3dff00",
    "#e6e6e6",
    "#bfc2c7",
    "#a6a6ab",
    "#8a99c7",
    "#9c7ac7",
    "#e06633",
    "#f090a0",
    "#50d050",
    "#c88033",
    "#7d80b0",
    "#c28f8f",
    "#668f8f",
    "#bd80e3",
    "#ffa100",
    "#a62929",
    "#5cb8d1",
    "#702eb0",
    "#00ff00",
    "#94ffff",
    "#94e0e0",
    "#73c2c9",
    "#54b5b5",
    "#3b9e9e",
    "#248f8f",
    "#0a7d8c",
    "#006985",
    "#c0c0c0",
    "#ffd98f",
    "#a67573",
    "#668080",
    "#9e63b5",
    "#d47a00",
    "#940094",
    "#429eb0",
    "#57178f",
    "#00c900",
    "#70d4ff",
    "#ffffc7",
    "#d9ffc7",
    "#c7ffc7",
    "#a3ffc7",
    "#8fffc7",
    "#61ffc7",
    "#45ffc7",
    "#30ffc7",
    "#1fffc7",
    "#00ff9c",
    "#00e675",
    "#00d452",
    "#00bf38",
    "#00ab24",
    "#4dc2ff",
    "#4da6ff",
    "#2194d6",
    "#267dab",
    "#266696",
    "#175487",
    "#d0d0e0",
    "#ffd123",
    "#b8b8d0",
    "#a6544d",
    "#575961",
    "#9e4fb5",
    "#ab5c00",
    "#754f45",
    "#428296",
    "#420066",
    "#007d00",
    "#70abfa",
    "#00baff",
    "#00a1ff",
    "#008fff",
    "#0080ff",
    "#006bff",
    "#545cf2",
    "#785ce3",
    "#8a4fe3",
    "#a136d4",
    "#b31fd4",
    "#b31fba",
    "#b30da6",
    "#bd0d87",
    "#c70066",
    "#cc0059",
    "#d1004f",
    "#d90045",
    "#e00038",
    "#e6002e",
    "#eb0026",
    "#f0001c",
    "#f50010",
    "#fa0004",
    "#ff0900",
    "#ff1a05",
    "#ff2b0a",
    "#ff3b0f",
    "#ff3f14",
    "#ff5b1a",
)


class Plot:
    """3D plot."""

    def __init__(self, size: float) -> None:
        if size <= 0:
            raise ValueError("non-positive size")
        ax = figure().add_subplot(projection="3d")
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)
        ax.set_zlim(-size, size)
        ax.set_aspect("equal")
        ax.set_axis_off()
        self._size = size
        self._ax = ax

    def point(
        self,
        pos: Vector,
        color: Optional[str] = None,
        size: float = 20.0,
    ) -> None:
        if size < 0:
            raise ValueError("negative size")
        self._ax.scatter(pos[0], pos[1], pos[2], color=color, s=size)

    def points(
        self,
        arr: Sequence[Point],
        color: Optional[str] = None,
        size: float = 20.0,
    ) -> None:
        if size < 0:
            raise ValueError("negative size")
        xs = []
        ys = []
        zs = []
        for elem in arr:
            pos = elem.pos
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
        self._ax.scatter(xs, ys, zs, color=color, s=size)

    def line(
        self, pos1: Vector, pos2: Vector, color: Optional[str] = None
    ) -> None:
        self._ax.plot(
            (pos1[0], pos2[0]),
            (pos1[1], pos2[1]),
            (pos1[2], pos2[2]),
            color=color,
        )

    def axis(self, direction: Vector, color: Optional[str] = None) -> None:
        direction = mul(normalize(direction), self._size)
        self.line(direction, neg(direction), color=color)

    def plane(
        self,
        normal: Vector,
        color: Optional[str] = None,
        opacity: float = 0.5,
    ) -> None:
        normal = normalize(normal)
        # `matplotlib` is unable to draw polygons parallel to the z-axis
        if abs(normal[2]) < _TOL:
            normal = normalize((normal[0], normal[1], choice((1, -1)) * _TOL))
        vec1 = cross(normal, PRIMAX)
        vec_norm = norm(vec1)
        if vec_norm == 0.0:
            vec1 = cross(normal, SECAX)
            vec_norm = norm(vec1)
        vec1 = mul(vec1, 1.0 / vec_norm)
        vec2 = normalize(cross(normal, vec1))
        size = self._size
        comps = ((size, size), (size, -size), (-size, -size), (-size, size))
        vertices = []
        for comp1, comp2 in comps:
            vertices.append(mul(vec1, comp1) + mul(vec2, comp2))
        fragments: Tuple[List[float], List[float], List[float]] = ([], [], [])
        for idx in range(len(vertices)):
            for i, (comp1, comp2) in enumerate(
                zip(vertices[idx - 1], vertices[idx])
            ):
                fragment = fragments[i]
                fragment.append(0)
                fragment.append(comp1)
                fragment.append(comp2)
        self._ax.plot_trisurf(
            fragments[0],
            fragments[1],
            fragments[2],
            color=color,
            alpha=opacity,
        )

    def molecule(
        self,
        arr: Sequence[LabeledPoint],
        size: float = 1.0,
        bond_tol: float = 0.02,
    ) -> None:
        colors = []
        radii = []
        for elem in arr:
            label = elem.label
            if label not in ELEM_SYMBS:
                raise ValueError(f"unknown element: {label}")
            idx = ELEM_SYMBS.index(elem.label)
            color = ELEM_COLORS[idx]
            colors.append(color)
            radius = ATOMIC_RADII[idx]
            radii.append(radius)
            self.point(elem.pos, color, size * radius)
        n_elems = len(arr)
        for i1 in range(n_elems - 1):
            pos1 = arr[i1].pos
            radius1 = radii[i1]
            for i2 in range(i1 + 1, n_elems):
                pos2 = arr[i2].pos
                radius2 = radii[i2]
                if norm(sub(pos1, pos2)) < bond_tol * (radius1 + radius2):
                    center = mul(add(pos1, pos2), 0.5)
                    self.line(pos1, center, colors[i1])
                    self.line(pos2, center, colors[i2])
