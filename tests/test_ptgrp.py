from .init import TestCase, main

from symmtools import (
    ptgrp,
    symmelems,
    chcoords,
    signvar,
    Points,
    generate,
    Point,
    Rotation,
    INF,
    PI,
    TOL,
    PHI,
    ax3permut,
)

point = Points.from_arr(chcoords([[]]))
segment = Points.from_arr(chcoords(signvar([1])))
collinear = Points.from_arr(chcoords(signvar([1])) + chcoords([[2]]))
triangle = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 2 * PI / 3)], TOL)
square = Points.from_arr(chcoords(signvar([1, 1])))
pentagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 2 * PI / 5)], TOL)
hexagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], PI / 3)], TOL)
tetrahedron = Points.from_arr(signvar([1, 1, 1], 1))
cube = Points.from_arr(signvar([1, 1, 1]))
octahedron = Points.from_arr(ax3permut(signvar([1])))
icosahedron = Points.from_arr(ax3permut(signvar([PHI, 1])))
dodecahedron = Points.from_arr(
    signvar([PHI, PHI, PHI]) + ax3permut(signvar([PHI + 1, 1]))
)


class TestPtGrp(TestCase):
    def test_symmelems(self) -> None:
        dim, inv, rots, refls, rotorefls = symmelems(point)
        self.assertEqual(dim, 0)
        self.assertTrue(inv)
        self.assertListEqual([rot.order for rot in rots], [])
        self.assertEqual(len(refls), 0)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [])
        dim, inv, rots, refls, rotorefls = symmelems(segment)
        self.assertEqual(dim, 1)
        self.assertTrue(inv)
        self.assertListEqual([rot.order for rot in rots], [INF])
        self.assertEqual(len(refls), 1)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [INF])
        dim, inv, rots, refls, rotorefls = symmelems(collinear)
        self.assertEqual(dim, 1)
        self.assertFalse(inv)
        self.assertListEqual([rot.order for rot in rots], [INF])
        self.assertEqual(len(refls), 0)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [])
        dim, inv, rots, refls, rotorefls = symmelems(triangle)
        n = 3
        self.assertEqual(dim, 2)
        self.assertFalse(inv)
        self.assertListEqual([rot.order for rot in rots], [n] + n * [2])
        self.assertEqual(len(refls), n + 1)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [n])
        dim, inv, rots, refls, rotorefls = symmelems(square)
        n = 4
        self.assertEqual(dim, 2)
        self.assertTrue(inv)
        self.assertListEqual([rot.order for rot in rots], [n] + n * [2])
        self.assertEqual(len(refls), n + 1)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [n])
        dim, inv, rots, refls, rotorefls = symmelems(pentagon)
        n = 5
        self.assertEqual(dim, 2)
        self.assertFalse(inv)
        self.assertListEqual([rot.order for rot in rots], [n] + n * [2])
        self.assertEqual(len(refls), n + 1)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [n])
        dim, inv, rots, refls, rotorefls = symmelems(hexagon)
        n = 6
        self.assertEqual(dim, 2)
        self.assertTrue(inv)
        self.assertListEqual([rot.order for rot in rots], [n] + n * [2])
        self.assertEqual(len(refls), n + 1)
        self.assertListEqual([rotorefl.order for rotorefl in rotorefls], [n])
        dim, inv, rots, refls, rotorefls = symmelems(tetrahedron)
        self.assertEqual(dim, 3)
        self.assertFalse(inv)
        self.assertListEqual([rot.order for rot in rots], 4 * [3] + 3 * [2])
        self.assertEqual(len(refls), 6)
        self.assertListEqual(
            [rotorefl.order for rotorefl in rotorefls], 3 * [4]
        )
        dim, inv, rots, refls, rotorefls = symmelems(cube)
        self.assertEqual(dim, 3)
        self.assertTrue(inv)
        self.assertListEqual(
            [rot.order for rot in rots], 3 * [4] + 4 * [3] + 6 * [2]
        )
        self.assertEqual(len(refls), 9)
        self.assertListEqual(
            [rotorefl.order for rotorefl in rotorefls], 4 * [6] + 3 * [4]
        )
        dim, inv, rots, refls, rotorefls = symmelems(octahedron)
        self.assertEqual(dim, 3)
        self.assertTrue(inv)
        self.assertListEqual(
            [rot.order for rot in rots], 3 * [4] + 4 * [3] + 6 * [2]
        )
        self.assertEqual(len(refls), 9)
        self.assertListEqual(
            [rotorefl.order for rotorefl in rotorefls], 4 * [6] + 3 * [4]
        )
        dim, inv, rots, refls, rotorefls = symmelems(icosahedron)
        self.assertEqual(dim, 3)
        self.assertTrue(inv)
        self.assertListEqual(
            [rot.order for rot in rots], 6 * [5] + 10 * [3] + 15 * [2]
        )
        self.assertEqual(len(refls), 15)
        self.assertListEqual(
            [rotorefl.order for rotorefl in rotorefls], 6 * [10] + 10 * [6]
        )
        dim, inv, rots, refls, rotorefls = symmelems(dodecahedron)
        self.assertEqual(dim, 3)
        self.assertTrue(inv)
        self.assertListEqual(
            [rot.order for rot in rots], 6 * [5] + 10 * [3] + 15 * [2]
        )
        self.assertEqual(len(refls), 15)
        # self.assertListEqual(
        #     [rotorefl.order for rotorefl in rotorefls], 6 * [10] + 10 * [6]
        # )

    def test_ptgrp(self) -> None:
        self.assertEqual(ptgrp(point), "Kh")
        self.assertEqual(ptgrp(segment), "Dooh")
        self.assertEqual(ptgrp(collinear), "Coov")
        self.assertEqual(ptgrp(triangle), "D3h")
        self.assertEqual(ptgrp(square), "D4h")
        self.assertEqual(ptgrp(pentagon), "D5h")
        self.assertEqual(ptgrp(hexagon), "D6h")
        self.assertEqual(ptgrp(tetrahedron), "Td")
        self.assertEqual(ptgrp(cube), "Oh")
        self.assertEqual(ptgrp(octahedron), "Oh")
        self.assertEqual(ptgrp(icosahedron), "Ih")
        self.assertEqual(ptgrp(dodecahedron), "Ih")


if __name__ == "__main__":
    main()
