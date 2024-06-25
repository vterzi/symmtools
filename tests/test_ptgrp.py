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
        asymmetric = Points.from_arr(
            [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]
        )
        symm_elem_symbs, group_symb = self.stringify(asymmetric)
        self.assertEqual(symm_elem_symbs, "")
        self.assertEqual(group_symb, "C1")

        plane = Points.from_arr(chcoords([[0, 0], [1, 0], [0, 2]]))
        symm_elem_symbs, group_symb = self.stringify(plane)
        self.assertEqual(symm_elem_symbs, "s")
        self.assertEqual(group_symb, "Cs")

        prism = Points.from_symm(
            chcoords([[0, 0], [1, 0], [0, 2]], [0, 0, 1]),
            [ReflectionPlane([0, 0, 1])],
        )
        symm_elem_symbs, group_symb = self.stringify(prism)
        self.assertEqual(symm_elem_symbs, "s")
        self.assertEqual(group_symb, "Cs")

        antiprism = Points.from_symm(
            chcoords([[0, 0], [1, 0], [0, 2]], [0, 0, 1]), [InversionCenter()]
        )
        symm_elem_symbs, group_symb = self.stringify(antiprism)
        self.assertEqual(symm_elem_symbs, "i")
        self.assertEqual(group_symb, "Ci")

    def test_var_from_all_symm_elems(self) -> None:
        # S(2(n=2k))   | Cn,S(2n)
        # S(2(n=2k+1)) | Cn,i,S(2n)

        two_fold = Points.from_symm(
            [[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 2)]
        )
        symm_elem_symbs, group_symb = self.stringify(two_fold)
        self.assertEqual(symm_elem_symbs, "C2")
        self.assertEqual(group_symb, "C2")

        three_fold = Points.from_symm(
            [[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 3)]
        )
        symm_elem_symbs, group_symb = self.stringify(three_fold)
        self.assertEqual(symm_elem_symbs, "C3")
        self.assertEqual(group_symb, "C3")

        four_fold = Points.from_symm(
            [[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 4)]
        )
        symm_elem_symbs, group_symb = self.stringify(four_fold)
        self.assertEqual(symm_elem_symbs, "C4")
        self.assertEqual(group_symb, "C4")

        five_fold = Points.from_symm(
            [[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 5)]
        )
        symm_elem_symbs, group_symb = self.stringify(five_fold)
        self.assertEqual(symm_elem_symbs, "C5")
        self.assertEqual(group_symb, "C5")

        six_fold = Points.from_symm(
            [[2, 0, 0], [2, 1, 1]], [RotationAxis([0, 0, 1], 6)]
        )
        symm_elem_symbs, group_symb = self.stringify(six_fold)
        self.assertEqual(symm_elem_symbs, "C6")
        self.assertEqual(group_symb, "C6")

        angle = Points.from_arr(
            chcoords([[]]) + chcoords(signvar([1]), [0, 0, 1])
        )
        symm_elem_symbs, group_symb = self.stringify(angle)
        self.assertEqual(symm_elem_symbs, "C2,2s")
        self.assertEqual(group_symb, "C2v")

        triangular_pyramid = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm([[1, 0, 1]], [RotationAxis([0, 0, 1], 3)])
        symm_elem_symbs, group_symb = self.stringify(triangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C3,3s")
        self.assertEqual(group_symb, "C3v")

        quadrangular_pyramid = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm([[1, 0, 1]], [RotationAxis([0, 0, 1], 4)])
        symm_elem_symbs, group_symb = self.stringify(quadrangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C4,4s")
        self.assertEqual(group_symb, "C4v")

        pentangular_pyramid = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm([[1, 0, 1]], [RotationAxis([0, 0, 1], 5)])
        symm_elem_symbs, group_symb = self.stringify(pentangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C5,5s")
        self.assertEqual(group_symb, "C5v")

        hexangular_pyramid = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm([[1, 0, 1]], [RotationAxis([0, 0, 1], 6)])
        symm_elem_symbs, group_symb = self.stringify(hexangular_pyramid)
        self.assertEqual(symm_elem_symbs, "C6,6s")
        self.assertEqual(group_symb, "C6v")

        propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
            [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 2)]
        )
        symm_elem_symbs, group_symb = self.stringify(propeller)
        self.assertEqual(symm_elem_symbs, "C2,s,i")
        self.assertEqual(group_symb, "C2h")

        triple_propeller = Points.from_arr(chcoords([[]])) + Points.from_symm(
            [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 3)]
        )
        symm_elem_symbs, group_symb = self.stringify(triple_propeller)
        self.assertEqual(symm_elem_symbs, "C3,s,S3")
        self.assertEqual(group_symb, "C3h")

        quadruple_propeller = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm(
            [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 4)]
        )
        symm_elem_symbs, group_symb = self.stringify(quadruple_propeller)
        self.assertEqual(symm_elem_symbs, "C4,s,i,S4")
        self.assertEqual(group_symb, "C4h")

        pentuple_propeller = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm(
            [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 5)]
        )
        symm_elem_symbs, group_symb = self.stringify(pentuple_propeller)
        self.assertEqual(symm_elem_symbs, "C5,s,S5")
        self.assertEqual(group_symb, "C5h")

        hextuple_propeller = Points.from_arr(
            chcoords([[]])
        ) + Points.from_symm(
            [[1, 0, 0], [2, 1, 0]], [RotationAxis([0, 0, 1], 6)]
        )
        symm_elem_symbs, group_symb = self.stringify(hextuple_propeller)
        self.assertEqual(symm_elem_symbs, "C6,s,i,S6")
        self.assertEqual(group_symb, "C6h")

        obj = Points.from_symm(
            [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 4)]
        )
        symm_elem_symbs, group_symb = self.stringify(obj)
        self.assertEqual(symm_elem_symbs, "C2,S4")
        self.assertEqual(group_symb, "S4")

        obj = Points.from_symm(
            [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 6)]
        )
        symm_elem_symbs, group_symb = self.stringify(obj)
        self.assertEqual(symm_elem_symbs, "C3,i,S6")
        self.assertEqual(group_symb, "S6")

        obj = Points.from_symm(
            [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 8)]
        )
        symm_elem_symbs, group_symb = self.stringify(obj)
        self.assertEqual(symm_elem_symbs, "C4,S8")
        self.assertEqual(group_symb, "S8")

        obj = Points.from_symm(
            [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 10)]
        )
        symm_elem_symbs, group_symb = self.stringify(obj)
        self.assertEqual(symm_elem_symbs, "C5,i,S10")
        self.assertEqual(group_symb, "S10")

        obj = Points.from_symm(
            [[1, 0, 2], [2, 1, 2]], [RotoreflectionAxis([0, 0, 1], 12)]
        )
        symm_elem_symbs, group_symb = self.stringify(obj)
        self.assertEqual(symm_elem_symbs, "C6,S12")
        self.assertEqual(group_symb, "S12")

        twist = Points.from_arr(
            chcoords(signvar([2, 1], 1), [0, 0, 1])
            + chcoords(signvar([1, 2], 1), [0, 0, -1])
        )
        symm_elem_symbs, group_symb = self.stringify(twist)
        self.assertEqual(symm_elem_symbs, "3C2")
        self.assertEqual(group_symb, "D2")

        triple_helix = Points.from_symm(
            [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 3)]
        )
        symm_elem_symbs, group_symb = self.stringify(triple_helix)
        self.assertEqual(symm_elem_symbs, "C3,3C2")
        self.assertEqual(group_symb, "D3")

        quadruple_helix = Points.from_symm(
            [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 4)]
        )
        symm_elem_symbs, group_symb = self.stringify(quadruple_helix)
        self.assertEqual(symm_elem_symbs, "C4,4C2")
        self.assertEqual(group_symb, "D4")

        pentuple_helix = Points.from_symm(
            [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 5)]
        )
        symm_elem_symbs, group_symb = self.stringify(pentuple_helix)
        self.assertEqual(symm_elem_symbs, "C5,5C2")
        self.assertEqual(group_symb, "D5")

        hextuple_helix = Points.from_symm(
            [[2, 1, 1], [2, -1, -1]], [RotationAxis([0, 0, 1], 6)]
        )
        symm_elem_symbs, group_symb = self.stringify(hextuple_helix)
        self.assertEqual(symm_elem_symbs, "C6,6C2")
        self.assertEqual(group_symb, "D6")

        quarter_twist = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 4)]
        )
        symm_elem_symbs, group_symb = self.stringify(quarter_twist)
        self.assertEqual(symm_elem_symbs, "3C2,2s,S4")
        self.assertEqual(group_symb, "D2d")

        triangular_antiprism = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 6)]
        )
        symm_elem_symbs, group_symb = self.stringify(triangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C3,3C2,3s,i,S6")
        self.assertEqual(group_symb, "D3d")

        quadrangular_antiprism = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 8)]
        )
        symm_elem_symbs, group_symb = self.stringify(quadrangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C4,4C2,4s,S8")
        self.assertEqual(group_symb, "D4d")

        pentangular_antiprism = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 10)]
        )
        symm_elem_symbs, group_symb = self.stringify(pentangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C5,5C2,5s,i,S10")
        self.assertEqual(group_symb, "D5d")

        hexangular_antiprism = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 12)]
        )
        symm_elem_symbs, group_symb = self.stringify(hexangular_antiprism)
        self.assertEqual(symm_elem_symbs, "C6,6C2,6s,S12")
        self.assertEqual(group_symb, "D6d")

        rectangle = Points.from_arr(chcoords(signvar([2, 1])))
        symm_elem_symbs, group_symb = self.stringify(rectangle)
        self.assertEqual(symm_elem_symbs, "3C2,3s,i")
        self.assertEqual(group_symb, "D2h")

        rectangular_prism = Points.from_arr(signvar([3, 2, 1]))
        symm_elem_symbs, group_symb = self.stringify(rectangular_prism)
        self.assertEqual(symm_elem_symbs, "3C2,3s,i")
        self.assertEqual(group_symb, "D2h")

        triangle = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 3)])
        symm_elem_symbs, group_symb = self.stringify(triangle)
        self.assertEqual(symm_elem_symbs, "C3,3C2,4s,S3")
        self.assertEqual(group_symb, "D3h")

        triangular_prism = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 3)]
        )
        symm_elem_symbs, group_symb = self.stringify(triangular_prism)
        self.assertEqual(symm_elem_symbs, "C3,3C2,4s,S3")
        self.assertEqual(group_symb, "D3h")

        square = Points.from_arr(chcoords(signvar([1, 1])))
        symm_elem_symbs, group_symb = self.stringify(square)
        self.assertEqual(symm_elem_symbs, "C4,4C2,5s,i,S4")
        self.assertEqual(group_symb, "D4h")

        quadrangular_prism = Points.from_symm(
            [[1, 0, 2]],
            [RotationAxis([0, 0, 1], 4), ReflectionPlane([0, 0, 1])],
        )
        symm_elem_symbs, group_symb = self.stringify(quadrangular_prism)
        self.assertEqual(symm_elem_symbs, "C4,4C2,5s,i,S4")
        self.assertEqual(group_symb, "D4h")

        pentagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 5)])
        symm_elem_symbs, group_symb = self.stringify(pentagon)
        self.assertEqual(symm_elem_symbs, "C5,5C2,6s,S5")
        self.assertEqual(group_symb, "D5h")

        pentangular_prism = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 5)]
        )
        symm_elem_symbs, group_symb = self.stringify(pentangular_prism)
        self.assertEqual(symm_elem_symbs, "C5,5C2,6s,S5")
        self.assertEqual(group_symb, "D5h")

        hexagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 6)])
        symm_elem_symbs, group_symb = self.stringify(hexagon)
        self.assertEqual(symm_elem_symbs, "C6,6C2,7s,i,S6")
        self.assertEqual(group_symb, "D6h")

        hexangular_prism = Points.from_symm(
            [[1, 0, 2]],
            [RotationAxis([0, 0, 1], 6), ReflectionPlane([0, 0, 1])],
        )
        symm_elem_symbs, group_symb = self.stringify(hexangular_prism)
        self.assertEqual(symm_elem_symbs, "C6,6C2,7s,i,S6")
        self.assertEqual(group_symb, "D6h")

    def test_high_from_all_symm_elems(self) -> None:
        tetrahedral = Points.from_arr(ax3permut(signvar([3, 2, 1], 1)))
        symm_elem_symbs, group_symb = self.stringify(tetrahedral)
        self.assertEqual(symm_elem_symbs, "4C3,3C2")
        self.assertEqual(group_symb, "T")

        tetrahedron = Points.from_arr(signvar([1, 1, 1], 1))
        symm_elem_symbs, group_symb = self.stringify(tetrahedron)
        self.assertEqual(symm_elem_symbs, "4C3,3C2,6s,3S4")
        self.assertEqual(group_symb, "Td")

        pyritohedron = Points.from_arr(ax3permut(signvar([2, 1])))
        symm_elem_symbs, group_symb = self.stringify(pyritohedron)
        self.assertEqual(symm_elem_symbs, "4C3,3C2,3s,i,4S6")
        self.assertEqual(group_symb, "Th")

        octahedral = Points.from_arr(
            ax3permut(signvar([3, 2, 1], 1))
            + ax3permut(signvar([2, 3, 1], -1))
        )
        symm_elem_symbs, group_symb = self.stringify(octahedral)
        self.assertEqual(symm_elem_symbs, "3C4,4C3,6C2")
        self.assertEqual(group_symb, "O")

        cube = Points.from_arr(signvar([1, 1, 1]))
        symm_elem_symbs, group_symb = self.stringify(cube)
        self.assertEqual(symm_elem_symbs, "3C4,4C3,6C2,9s,i,4S6,3S4")
        self.assertEqual(group_symb, "Oh")

        octahedron = Points.from_arr(ax3permut(signvar([1])))
        symm_elem_symbs, group_symb = self.stringify(octahedron)
        self.assertEqual(symm_elem_symbs, "3C4,4C3,6C2,9s,i,4S6,3S4")
        self.assertEqual(group_symb, "Oh")

        # icosahedral = Points.from_arr(...)
        # symm_elem_symbs, group_symb = self.stringify(icosahedral)
        # self.assertEqual(symm_elem_symbs, "6C5,10C3,15C2")
        # self.assertEqual(group_symb, "I")

        icosahedron = Points.from_arr(ax3permut(signvar([PHI, 1])))
        symm_elem_symbs, group_symb = self.stringify(icosahedron)
        self.assertEqual(symm_elem_symbs, "6C5,10C3,15C2,15s,i,6S10,10S6")
        self.assertEqual(group_symb, "Ih")

        dodecahedron = Points.from_arr(
            signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
        )
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

        asymmetric_line = Points.from_arr(
            chcoords(signvar([1])) + chcoords([[2]])
        )
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

        symmetric_line = Points.from_arr(chcoords(signvar([1])))
        symm_elem_symbs, group_symb = self.stringify(symmetric_line)
        self.assertEqual(symm_elem_symbs, "Coo,ooC2,oosv,s,i,Soo")
        self.assertEqual(group_symb, "Dooh")

        symm_elems = [CenterRotationAxes()]
        symm_elem_symbs, group_symb = self.stringify(symm_elems)
        self.assertEqual(symm_elem_symbs, "ooCoo")
        self.assertEqual(group_symb, "K")

        point = Points.from_arr(chcoords([[]]))
        symm_elem_symbs, group_symb = self.stringify(point)
        self.assertEqual(symm_elem_symbs, "ooCoo,oos,i,ooSoo")
        self.assertEqual(group_symb, "Kh")


if __name__ == "__main__":
    main()
