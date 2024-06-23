from .init import TestCase, main, Tuple

from symmtools import (
    chcoords,
    signvar,
    ax3permut,
    Points,
    RotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    PointGroup,
    SymmetryElements,
    PHI,
    TOL,
)


class TestPtGrp(TestCase):
    def test_symm_elems(self) -> None:
        def init(points: Points) -> Tuple[str, str]:
            symm_elems = tuple(points.symm_elems(TOL))
            info = SymmetryElements()
            info.include(symm_elems, TOL)
            group = PointGroup.from_all_symm_elems(symm_elems)
            return ",".join(info.symbs), group.symb

        triangle = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 3)])
        symbs, symb = init(triangle)
        self.assertEqual(symbs, "C3,3C2,4s,S3")
        self.assertEqual(symb, "D3h")

        ecliptic_triangles = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 3)]
        )
        symbs, symb = init(ecliptic_triangles)
        self.assertEqual(symbs, "C3,3C2,4s,S3")
        self.assertEqual(symb, "D3h")

        staggered_triangles = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 6)]
        )
        symbs, symb = init(staggered_triangles)
        self.assertEqual(symbs, "C3,3C2,3s,i,S6")
        self.assertEqual(symb, "D3d")

        square = Points.from_arr(chcoords(signvar([1, 1])))
        symbs, symb = init(square)
        self.assertEqual(symbs, "C4,4C2,5s,i,S4")
        self.assertEqual(symb, "D4h")

        ecliptic_squares = Points.from_symm(
            [[1, 0, 2]],
            [RotationAxis([0, 0, 1], 4), ReflectionPlane([0, 0, 1])],
        )
        symbs, symb = init(ecliptic_squares)
        self.assertEqual(symbs, "C4,4C2,5s,i,S4")
        self.assertEqual(symb, "D4h")

        staggered_squares = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 8)]
        )
        symbs, symb = init(staggered_squares)
        self.assertEqual(symbs, "C4,4C2,4s,S8")
        self.assertEqual(symb, "D4d")

        pentagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 5)])
        symbs, symb = init(pentagon)
        self.assertEqual(symbs, "C5,5C2,6s,S5")
        self.assertEqual(symb, "D5h")

        ecliptic_pentagons = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 5)]
        )
        symbs, symb = init(ecliptic_pentagons)
        self.assertEqual(symbs, "C5,5C2,6s,S5")
        self.assertEqual(symb, "D5h")

        staggered_pentagons = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 10)]
        )
        symbs, symb = init(staggered_pentagons)
        self.assertEqual(symbs, "C5,5C2,5s,i,S10")
        self.assertEqual(symb, "D5d")

        hexagon = Points.from_symm([[1, 0, 0]], [RotationAxis([0, 0, 1], 6)])
        symbs, symb = init(hexagon)
        self.assertEqual(symbs, "C6,6C2,7s,i,S6")
        self.assertEqual(symb, "D6h")

        ecliptic_hexagons = Points.from_symm(
            [[1, 0, 2]],
            [RotationAxis([0, 0, 1], 6), ReflectionPlane([0, 0, 1])],
        )
        symbs, symb = init(ecliptic_hexagons)
        self.assertEqual(symbs, "C6,6C2,7s,i,S6")
        self.assertEqual(symb, "D6h")

        staggered_hexagons = Points.from_symm(
            [[1, 0, 2]], [RotoreflectionAxis([0, 0, 1], 12)]
        )
        symbs, symb = init(staggered_hexagons)
        self.assertEqual(symbs, "C6,6C2,6s,S12")
        self.assertEqual(symb, "D6d")

        tetrahedron = Points.from_arr(signvar([1, 1, 1], 1))
        symbs, symb = init(tetrahedron)
        self.assertEqual(symbs, "4C3,3C2,6s,3S4")
        self.assertEqual(symb, "Td")

        cube = Points.from_arr(signvar([1, 1, 1]))
        symbs, symb = init(cube)
        self.assertEqual(symbs, "3C4,4C3,6C2,9s,i,4S6,3S4")
        self.assertEqual(symb, "Oh")

        octahedron = Points.from_arr(ax3permut(signvar([1])))
        symbs, symb = init(octahedron)
        self.assertEqual(symbs, "3C4,4C3,6C2,9s,i,4S6,3S4")
        self.assertEqual(symb, "Oh")

        icosahedron = Points.from_arr(ax3permut(signvar([PHI, 1])))
        symbs, symb = init(icosahedron)
        self.assertEqual(symbs, "6C5,10C3,15C2,15s,i,6S10,10S6")
        self.assertEqual(symb, "Ih")

        dodecahedron = Points.from_arr(
            signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
        )
        symbs, symb = init(dodecahedron)
        self.assertEqual(symbs, "6C5,10C3,15C2,15s,i,6S10,10S6")
        self.assertEqual(symb, "Ih")

        collinear = Points.from_arr(chcoords(signvar([1])) + chcoords([[2]]))
        symbs, symb = init(collinear)
        self.assertEqual(symbs, "Coo,oosv")
        self.assertEqual(symb, "Coov")

        segment = Points.from_arr(chcoords(signvar([1])))
        symbs, symb = init(segment)
        self.assertEqual(symbs, "Coo,ooC2,oosv,s,i,Soo")
        self.assertEqual(symb, "Dooh")

        point = Points.from_arr(chcoords([[]]))
        symbs, symb = init(point)
        self.assertEqual(symbs, "ooCoo,oos,i,ooSoo")
        self.assertEqual(symb, "Kh")


if __name__ == "__main__":
    main()
