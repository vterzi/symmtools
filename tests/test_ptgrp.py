from .init import TestCase, main, Union, Sequence, Tuple

from symmtools import (
    chcoords,
    signvar,
    ax3permut,
    Points,
    SymmetryElement,
    InversionCenter,
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
    AxisRotationAxes,
    CenterRotationAxes,
    PointGroup,
    SymmetryElements,
    PHI,
    TOL,
)


asymmetric = Points.from_arr([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
plane = Points.from_arr(chcoords([[0, 0], [1, 0], [0, 2]]))
prism = Points.from_symm(
    chcoords([[0, 0], [1, 0], [0, 2]], [0, 0, 1]),
    [ReflectionPlane([0, 0, 1])],
)
antiprism = Points.from_symm(
    chcoords([[0, 0], [1, 0], [0, 2]], [0, 0, 1]), [InversionCenter()]
)
rot2 = Points.from_symm([[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 2)])
rot3 = Points.from_symm([[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 3)])
rot4 = Points.from_symm([[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 4)])
rot5 = Points.from_symm([[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 5)])
rot6 = Points.from_symm([[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 6)])
angle = Points.from_arr(chcoords([[]]) + chcoords(signvar([1]), [0, 0, 1]))
triangular_pyramid = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 1]], [RotationAxis([0, 0, 1], 3)]
)
quadrangular_pyramid = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 1]], [RotationAxis([0, 0, 1], 4)]
)
pentangular_pyramid = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 1]], [RotationAxis([0, 0, 1], 5)]
)
hexangular_pyramid = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 1]], [RotationAxis([0, 0, 1], 6)]
)
propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 2)]
)
triple_propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 3)]
)
quadruple_propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 4)]
)
pentuple_propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 5)]
)
hextuple_propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
    [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 6)]
)
rotorefl4 = Points.from_symm(
    [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 4)]
)
rotorefl6 = Points.from_symm(
    [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 6)]
)
rotorefl8 = Points.from_symm(
    [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 8)]
)
rotorefl10 = Points.from_symm(
    [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 10)]
)
rotorefl12 = Points.from_symm(
    [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 12)]
)
twist = Points.from_arr(
    chcoords(signvar([2, 1], 1), [0, 0, 1])
    + chcoords(signvar([1, 2], 1), [0, 0, -1])
)
triple_helix = Points.from_symm(
    [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 3)]
)
quadruple_helix = Points.from_symm(
    [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 4)]
)
pentuple_helix = Points.from_symm(
    [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 5)]
)
hextuple_helix = Points.from_symm(
    [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 6)]
)
quarter_twist = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 4)]
)
triangular_antiprism = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 6)]
)
quadrangular_antiprism = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 8)]
)
pentangular_antiprism = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 10)]
)
hexangular_antiprism = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 12)]
)
rectangle = Points.from_arr(chcoords(signvar([2, 1])))
rectangular_prism = Points.from_arr(signvar([3, 2, 1]))
triangle = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 3)])
triangular_prism = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 3)]
)
square = Points.from_arr(chcoords(signvar([1, 1])))
quadrangular_prism = Points.from_symm(
    [[1, 0, 2]],
    [RotationAxis([0, 0, 1], 4), ReflectionPlane([0, 0, 1])],
)
pentagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 5)])
pentangular_prism = Points.from_symm(
    [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 5)]
)
hexagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 6)])
hexangular_prism = Points.from_symm(
    [[1, 0, 2]],
    [RotationAxis([0, 0, 1], 6), ReflectionPlane([0, 0, 1])],
)
tetrahedral = Points.from_arr(ax3permut(signvar([3, 2, 1], 1)))
tetrahedron = Points.from_arr(signvar([1, 1, 1], 1))
pyritohedron = Points.from_arr(ax3permut(signvar([2, 1])))
octahedral = Points.from_arr(
    ax3permut(signvar([3, 2, 1], 1)) + ax3permut(signvar([2, 3, 1], -1))
)
cube = Points.from_arr(signvar([1, 1, 1]))
octahedron = Points.from_arr(ax3permut(signvar([1])))
icosahedron = Points.from_arr(ax3permut(signvar([PHI, 1])))
dodecahedron = Points.from_arr(
    signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
)
asymmetric_line = Points.from_arr(chcoords(signvar([1])) + chcoords([[2]]))
symmetric_line = Points.from_arr(chcoords(signvar([1])))
point = Points.from_arr(chcoords([[]]))


class TestPointGroup(TestCase):
    def stringify(
        self, symm_elems: Union[Sequence[SymmetryElement], Points]
    ) -> Tuple[str, str]:
        if isinstance(symm_elems, Points):
            symm_elems = tuple(symm_elems.symm_elems(TOL))
        info = SymmetryElements()
        info.include(symm_elems, TOL)
        group = PointGroup.from_all_symm_elems(symm_elems)
        return ",".join(info.symbs), group.symb

    def test_low_from_all_symm_elems(self) -> None:
        symm_elem_symbs, group_symb = self.stringify(asymmetric)
        self.assertEqual(symm_elem_symbs, "")
        self.assertEqual(group_symb, "C1")

        symm_elem_symbs, group_symb = self.stringify(plane)
        self.assertEqual(symm_elem_symbs, "s")
        self.assertEqual(group_symb, "Cs")

        symm_elem_symbs, group_symb = self.stringify(prism)
        self.assertEqual(symm_elem_symbs, "s")
        self.assertEqual(group_symb, "Cs")

        symm_elem_symbs, group_symb = self.stringify(antiprism)
        self.assertEqual(symm_elem_symbs, "i")
        self.assertEqual(group_symb, "Ci")

    def test_var_from_all_symm_elems(self) -> None:
        symm_elem_symbs, group_symb = self.stringify(rot2)
        self.assertEqual(symm_elem_symbs, "C2")
        self.assertEqual(group_symb, "C2")

        symm_elem_symbs, group_symb = self.stringify(rot3)
        self.assertEqual(symm_elem_symbs, "C3")
        self.assertEqual(group_symb, "C3")

        symm_elem_symbs, group_symb = self.stringify(rot4)
        self.assertEqual(symm_elem_symbs, "C4")
        self.assertEqual(group_symb, "C4")

        symm_elem_symbs, group_symb = self.stringify(rot5)
        self.assertEqual(symm_elem_symbs, "C5")
        self.assertEqual(group_symb, "C5")

        symm_elem_symbs, group_symb = self.stringify(rot6)
        self.assertEqual(symm_elem_symbs, "C6")
        self.assertEqual(group_symb, "C6")

        symm_elem_symbs, group_symb = self.stringify(angle)
        self.assertEqual(symm_elem_symbs, "C2,2s")
        self.assertEqual(group_symb, "C2v")

        symm_elem_symbs, group_symb = self.stringify(triangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C3,3s")
        self.assertEqual(group_symb, "C3v")

        symm_elem_symbs, group_symb = self.stringify(quadrangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C4,4s")
        self.assertEqual(group_symb, "C4v")

        symm_elem_symbs, group_symb = self.stringify(pentangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C5,5s")
        self.assertEqual(group_symb, "C5v")

        symm_elem_symbs, group_symb = self.stringify(hexangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C6,6s")
        self.assertEqual(group_symb, "C6v")

        symm_elem_symbs, group_symb = self.stringify(propeller)
        self.assertEqual(symm_elem_symbs, "C2,s,i")
        self.assertEqual(group_symb, "C2h")

        symm_elem_symbs, group_symb = self.stringify(triple_propeller)
        self.assertEqual(symm_elem_symbs, "C3,s,S3")
        self.assertEqual(group_symb, "C3h")

        symm_elem_symbs, group_symb = self.stringify(quadruple_propeller)
        self.assertEqual(symm_elem_symbs, "C4,s,i,S4")
        self.assertEqual(group_symb, "C4h")

        symm_elem_symbs, group_symb = self.stringify(pentuple_propeller)
        self.assertEqual(symm_elem_symbs, "C5,s,S5")
        self.assertEqual(group_symb, "C5h")

        symm_elem_symbs, group_symb = self.stringify(hextuple_propeller)
        self.assertEqual(symm_elem_symbs, "C6,s,i,S6")
        self.assertEqual(group_symb, "C6h")

        symm_elem_symbs, group_symb = self.stringify(rotorefl4)
        self.assertEqual(symm_elem_symbs, "C2,S4")
        self.assertEqual(group_symb, "S4")

        symm_elem_symbs, group_symb = self.stringify(rotorefl6)
        self.assertEqual(symm_elem_symbs, "C3,i,S6")
        self.assertEqual(group_symb, "S6")

        symm_elem_symbs, group_symb = self.stringify(rotorefl8)
        self.assertEqual(symm_elem_symbs, "C4,S8")
        self.assertEqual(group_symb, "S8")

        symm_elem_symbs, group_symb = self.stringify(rotorefl10)
        self.assertEqual(symm_elem_symbs, "C5,i,S10")
        self.assertEqual(group_symb, "S10")

        symm_elem_symbs, group_symb = self.stringify(rotorefl12)
        self.assertEqual(symm_elem_symbs, "C6,S12")
        self.assertEqual(group_symb, "S12")

        symm_elem_symbs, group_symb = self.stringify(twist)
        self.assertEqual(symm_elem_symbs, "3C2")
        self.assertEqual(group_symb, "D2")

        symm_elem_symbs, group_symb = self.stringify(triple_helix)
        self.assertEqual(symm_elem_symbs, "C3,3C2")
        self.assertEqual(group_symb, "D3")

        symm_elem_symbs, group_symb = self.stringify(quadruple_helix)
        self.assertEqual(symm_elem_symbs, "C4,4C2")
        self.assertEqual(group_symb, "D4")

        symm_elem_symbs, group_symb = self.stringify(pentuple_helix)
        self.assertEqual(symm_elem_symbs, "C5,5C2")
        self.assertEqual(group_symb, "D5")

        symm_elem_symbs, group_symb = self.stringify(hextuple_helix)
        self.assertEqual(symm_elem_symbs, "C6,6C2")
        self.assertEqual(group_symb, "D6")

        symm_elem_symbs, group_symb = self.stringify(quarter_twist)
        self.assertEqual(symm_elem_symbs, "3C2,2s,S4")
        self.assertEqual(group_symb, "D2d")

        symm_elem_symbs, group_symb = self.stringify(triangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C3,3C2,3s,i,S6")
        self.assertEqual(group_symb, "D3d")

        symm_elem_symbs, group_symb = self.stringify(quadrangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C4,4C2,4s,S8")
        self.assertEqual(group_symb, "D4d")

        symm_elem_symbs, group_symb = self.stringify(pentangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C5,5C2,5s,i,S10")
        self.assertEqual(group_symb, "D5d")

        symm_elem_symbs, group_symb = self.stringify(hexangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C6,6C2,6s,S12")
        self.assertEqual(group_symb, "D6d")

        symm_elem_symbs, group_symb = self.stringify(rectangle)
        self.assertEqual(symm_elem_symbs, "3C2,3s,i")
        self.assertEqual(group_symb, "D2h")

        symm_elem_symbs, group_symb = self.stringify(rectangular_prism)
        self.assertEqual(symm_elem_symbs, "3C2,3s,i")
        self.assertEqual(group_symb, "D2h")

        symm_elem_symbs, group_symb = self.stringify(triangle)
        self.assertEqual(symm_elem_symbs, "C3,3C2,4s,S3")
        self.assertEqual(group_symb, "D3h")

        symm_elem_symbs, group_symb = self.stringify(triangular_prism)
        self.assertEqual(symm_elem_symbs, "C3,3C2,4s,S3")
        self.assertEqual(group_symb, "D3h")

        symm_elem_symbs, group_symb = self.stringify(square)
        self.assertEqual(symm_elem_symbs, "C4,4C2,5s,i,S4")
        self.assertEqual(group_symb, "D4h")

        symm_elem_symbs, group_symb = self.stringify(quadrangular_prism)
        self.assertEqual(symm_elem_symbs, "C4,4C2,5s,i,S4")
        self.assertEqual(group_symb, "D4h")

        symm_elem_symbs, group_symb = self.stringify(pentagon)
        self.assertEqual(symm_elem_symbs, "C5,5C2,6s,S5")
        self.assertEqual(group_symb, "D5h")

        symm_elem_symbs, group_symb = self.stringify(pentangular_prism)
        self.assertEqual(symm_elem_symbs, "C5,5C2,6s,S5")
        self.assertEqual(group_symb, "D5h")

        symm_elem_symbs, group_symb = self.stringify(hexagon)
        self.assertEqual(symm_elem_symbs, "C6,6C2,7s,i,S6")
        self.assertEqual(group_symb, "D6h")

        symm_elem_symbs, group_symb = self.stringify(hexangular_prism)
        self.assertEqual(symm_elem_symbs, "C6,6C2,7s,i,S6")
        self.assertEqual(group_symb, "D6h")

    def test_high_from_all_symm_elems(self) -> None:
        symm_elem_symbs, group_symb = self.stringify(tetrahedral)
        self.assertEqual(symm_elem_symbs, "4C3,3C2")
        self.assertEqual(group_symb, "T")

        symm_elem_symbs, group_symb = self.stringify(tetrahedron)
        self.assertEqual(symm_elem_symbs, "4C3,3C2,6s,3S4")
        self.assertEqual(group_symb, "Td")

        symm_elem_symbs, group_symb = self.stringify(pyritohedron)
        self.assertEqual(symm_elem_symbs, "4C3,3C2,3s,i,4S6")
        self.assertEqual(group_symb, "Th")

        symm_elem_symbs, group_symb = self.stringify(octahedral)
        self.assertEqual(symm_elem_symbs, "3C4,4C3,6C2")
        self.assertEqual(group_symb, "O")

        symm_elem_symbs, group_symb = self.stringify(cube)
        self.assertEqual(symm_elem_symbs, "3C4,4C3,6C2,9s,i,4S6,3S4")
        self.assertEqual(group_symb, "Oh")

        symm_elem_symbs, group_symb = self.stringify(octahedron)
        self.assertEqual(symm_elem_symbs, "3C4,4C3,6C2,9s,i,4S6,3S4")
        self.assertEqual(group_symb, "Oh")

        # icosahedral = Points.from_arr(...)
        # symm_elem_symbs, group_symb = self.stringify(icosahedral)
        # self.assertEqual(symm_elem_symbs, "6C5,10C3,15C2")
        # self.assertEqual(group_symb, "I")

        symm_elem_symbs, group_symb = self.stringify(icosahedron)
        self.assertEqual(symm_elem_symbs, "6C5,10C3,15C2,15s,i,6S10,10S6")
        self.assertEqual(group_symb, "Ih")

        symm_elem_symbs, group_symb = self.stringify(dodecahedron)
        self.assertEqual(symm_elem_symbs, "6C5,10C3,15C2,15s,i,6S10,10S6")
        self.assertEqual(group_symb, "Ih")

    def test_inf_from_all_symm_elems(self) -> None:
        symm_elems: Sequence[SymmetryElement]

        symm_elems = [InfRotationAxis([0, 0, 1])]
        symm_elem_symbs, group_symb = self.stringify(symm_elems)
        self.assertEqual(symm_elem_symbs, "Coo")
        self.assertEqual(group_symb, "Coo")

        symm_elems = [
            InfRotationAxis([0, 0, 1]),
            ReflectionPlane([0, 0, 1]),
            InversionCenter(),
            InfRotoreflectionAxis([0, 0, 1]),
        ]
        symm_elem_symbs, group_symb = self.stringify(symm_elems)
        self.assertEqual(symm_elem_symbs, "Coo,s,i,Soo")
        self.assertEqual(group_symb, "Cooh")

        symm_elem_symbs, group_symb = self.stringify(asymmetric_line)
        self.assertEqual(symm_elem_symbs, "Coo,oosv")
        self.assertEqual(group_symb, "Coov")

        symm_elems = [
            InfRotationAxis([0, 0, 1]),
            AxisRotationAxes([0, 0, 1]),
        ]
        symm_elem_symbs, group_symb = self.stringify(symm_elems)
        self.assertEqual(symm_elem_symbs, "Coo,ooC2")
        self.assertEqual(group_symb, "Doo")

        symm_elem_symbs, group_symb = self.stringify(symmetric_line)
        self.assertEqual(symm_elem_symbs, "Coo,ooC2,oosv,s,i,Soo")
        self.assertEqual(group_symb, "Dooh")

        symm_elems = [CenterRotationAxes()]
        symm_elem_symbs, group_symb = self.stringify(symm_elems)
        self.assertEqual(symm_elem_symbs, "ooCoo")
        self.assertEqual(group_symb, "K")

        symm_elem_symbs, group_symb = self.stringify(point)
        self.assertEqual(symm_elem_symbs, "ooCoo,oos,i,ooSoo")
        self.assertEqual(group_symb, "Kh")


if __name__ == "__main__":
    main()
