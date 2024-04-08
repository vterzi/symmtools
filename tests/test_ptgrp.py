from unittest import TestCase, main

from symmtools import (
    ptgrp,
    chcoords,
    signvar,
    topoints,
    Elems,
    generate,
    Point,
    Rotation,
    TOL,
    PHI,
    ax3permut,
)


class TestPtgrp(TestCase):
    def test_ptgrp(self):
        point = chcoords([[]])
        self.assertEqual(ptgrp(Elems(topoints(point))), "Kh")
        segment = chcoords(signvar([1]))
        self.assertEqual(ptgrp(Elems(topoints(segment))), "Dooh")
        collinear = segment + chcoords([[2]])
        self.assertEqual(ptgrp(Elems(topoints(collinear))), "Coov")
        triangle = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 3)], TOL)
        self.assertEqual(ptgrp(triangle), "D3h")
        square = Elems(topoints(chcoords(signvar([1, 1]))))
        self.assertEqual(ptgrp(square), "D4h")
        pentagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 5)], TOL)
        self.assertEqual(ptgrp(pentagon), "D5h")
        hexagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 6)], TOL)
        self.assertEqual(ptgrp(hexagon), "D6h")
        tetrahedron = Elems(topoints(signvar([1, 1, 1], 1)))
        self.assertEqual(ptgrp(tetrahedron), "Td")
        cube = Elems(topoints(signvar([1, 1, 1])))
        self.assertEqual(ptgrp(cube), "Oh")
        octahedron = Elems(topoints(ax3permut(signvar([1]))))
        self.assertEqual(ptgrp(octahedron), "Oh")
        icosahedron = Elems(topoints(ax3permut(signvar([PHI, 1]))))
        self.assertEqual(ptgrp(icosahedron), "Ih")
        dodecahedron = Elems(
            topoints(
                signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
            )
        )
        self.assertEqual(ptgrp(dodecahedron), "Ih")


if __name__ == "__main__":
    main()
