from .init import TestCase, main

from symmtools import (
    chcoords,
    signvar,
    ax3permut,
    Points,
    RotationAxis,
    PointGroup,
    SymmetryElements,
    PHI,
    TOL,
)


class TestPtGrp(TestCase):
    def test_symmelems(self) -> None:
        point = Points.from_arr(chcoords([[]]))
        symmelems = point.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "ooCoo,oos,i,ooSoo")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Kh")

        segment = Points.from_arr(chcoords(signvar([1])))
        symmelems = segment.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "Coo,ooC2,oosv,s,i,Soo")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Dooh")

        collinear = Points.from_arr(chcoords(signvar([1])) + chcoords([[2]]))
        symmelems = collinear.symmelems(TOL)
        print(symmelems)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "Coo,oosv")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Coov")

        triangle = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 3)])
        symmelems = triangle.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "C3,3C2,4s,S3")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "D3h")

        square = Points.from_arr(chcoords(signvar([1, 1])))
        symmelems = square.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "C4,4C2,5s,i,S4")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "D4h")

        pentagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 5)])
        symmelems = pentagon.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "C5,5C2,6s,S5")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "D5h")

        hexagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 6)])
        symmelems = hexagon.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "C6,6C2,7s,i,S6")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "D6h")

        tetrahedron = Points.from_arr(signvar([1, 1, 1], 1))
        symmelems = tetrahedron.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "4C3,3C2,6s,3S4")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Td")

        cube = Points.from_arr(signvar([1, 1, 1]))
        symmelems = cube.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "3C4,4C3,6C2,9s,i,4S6,3S4")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Oh")

        octahedron = Points.from_arr(ax3permut(signvar([1])))
        symmelems = octahedron.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "3C4,4C3,6C2,9s,i,4S6,3S4")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Oh")

        icosahedron = Points.from_arr(ax3permut(signvar([PHI, 1])))
        symmelems = icosahedron.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "6C5,10C3,15C2,15s,i,6S10,10S6")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Ih")

        dodecahedron = Points.from_arr(
            signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
        )
        symmelems = dodecahedron.symmelems(TOL)
        info = SymmetryElements()
        info.include(symmelems, TOL)
        self.assertEqual(",".join(info.symbs), "6C5,10C3,15C2,15s,i,6S10,10S6")
        group = PointGroup.from_all_symmelems(symmelems)
        self.assertEqual(group.symb, "Ih")


if __name__ == "__main__":
    main()
