from .init import TestCase, main

from symmtools import (
    ptgrp,
    chcoords,
    signvar,
    topoints,
    Points,
    generate,
    Point,
    Rotation,
    PI,
    TOL,
    PHI,
    ax3permut,
)


class TestPtGrp(TestCase):
    def test_ptgrp(self):
        point = chcoords([[]])
        self.assertEqual(ptgrp(Points(topoints(point))), "Kh")
        segment = chcoords(signvar([1]))
        self.assertEqual(ptgrp(Points(topoints(segment))), "Dooh")
        collinear = segment + chcoords([[2]])
        self.assertEqual(ptgrp(Points(topoints(collinear))), "Coov")
        triangle = generate(
            [Point([1, 0, 0])], [Rotation([0, 0, 1], 2 * PI / 3)], TOL
        )
        self.assertEqual(ptgrp(triangle), "D3h")
        square = Points(topoints(chcoords(signvar([1, 1]))))
        self.assertEqual(ptgrp(square), "D4h")
        pentagon = generate(
            [Point([1, 0, 0])], [Rotation([0, 0, 1], 2 * PI / 5)], TOL
        )
        self.assertEqual(ptgrp(pentagon), "D5h")
        hexagon = generate(
            [Point([1, 0, 0])], [Rotation([0, 0, 1], PI / 3)], TOL
        )
        self.assertEqual(ptgrp(hexagon), "D6h")
        tetrahedron = Points(topoints(signvar([1, 1, 1], 1)))
        self.assertEqual(ptgrp(tetrahedron), "Td")
        cube = Points(topoints(signvar([1, 1, 1])))
        self.assertEqual(ptgrp(cube), "Oh")
        octahedron = Points(topoints(ax3permut(signvar([1]))))
        self.assertEqual(ptgrp(octahedron), "Oh")
        icosahedron = Points(topoints(ax3permut(signvar([PHI, 1]))))
        self.assertEqual(ptgrp(icosahedron), "Ih")
        dodecahedron = Points(
            topoints(
                signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
            )
        )
        self.assertEqual(ptgrp(dodecahedron), "Ih")


if __name__ == "__main__":
    main()
