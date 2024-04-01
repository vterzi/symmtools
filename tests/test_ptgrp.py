from unittest import TestCase, main

from symmtools import ptgrp, project, signpermut, topoints, Elems, generate, Point, Rotation, TOL, PHI, ax3permut


class TestPtgrp(TestCase):
    def test_ptgrp(self):
        point = project([[]], 3*[0])
        self.assertEqual(ptgrp(Elems(topoints(point))), 'Kh')
        segment = project(signpermut([1]), 3*[0])
        self.assertEqual(ptgrp(Elems(topoints(segment))), 'Dooh')
        collinear = segment + project([[2]], 3*[0])
        self.assertEqual(ptgrp(Elems(topoints(collinear))), 'Coov')
        triangle = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 3)], TOL)
        self.assertEqual(ptgrp(triangle), 'D3h')
        square = Elems(topoints(project(signpermut([1, 1]), 3 * [0])))
        self.assertEqual(ptgrp(square), 'D4h')
        pentagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 5)], TOL)
        self.assertEqual(ptgrp(pentagon), 'D5h')
        hexagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 6)], TOL)
        self.assertEqual(ptgrp(hexagon), 'D6h')
        tetrahedron = Elems(topoints(signpermut([1, 1, 1], 1)))
        self.assertEqual(ptgrp(tetrahedron), 'Td')
        cube = Elems(topoints(signpermut([1, 1, 1])))
        self.assertEqual(ptgrp(cube), 'Oh')
        octahedron = Elems(topoints(ax3permut(signpermut([1]))))
        self.assertEqual(ptgrp(octahedron), 'Oh')
        icosahedron = Elems(topoints(ax3permut(signpermut([PHI, 1]))))
        self.assertEqual(ptgrp(icosahedron), 'Ih')
        dodecahedron = Elems(topoints(signpermut([PHI, PHI, PHI]) + ax3permut(signpermut([PHI + 1, 1]))))
        self.assertEqual(ptgrp(dodecahedron), 'Ih')


if __name__ == '__main__':
    main()
