from unittest import TestCase, main
from random import random, choice, randrange, randint, normalvariate
from typing import Tuple

from numpy import pi, array, zeros, sin, cos, cross, ndarray, float64
from numpy.typing import NDArray

from symmtools import (
    TOL,
    vector,
    canon,
    normalize,
    diff,
    same,
    indep,
    parallel,
    perpendicular,
    translate,
    invert,
    move2,
    rotate,
    reflect,
)


def randvec() -> NDArray[float64]:
    return array([normalvariate(0, 1) for _ in range(3)])


def randnonzerovec() -> Tuple[NDArray[float64], float64]:
    norm = float64(0)
    while norm == 0:
        vec = randvec()
        norm = (vec**2).sum() ** 0.5
    return vec, norm


def randunitvec() -> NDArray[float64]:
    vec, norm = randnonzerovec()
    return vec / norm


def perturb() -> NDArray[float64]:
    vec = zeros(3)
    vec[randrange(3)] = choice([-1, 1]) * TOL
    return vec


class TestVecOp(TestCase):
    def test_vector(self):
        arr = [randint(-8, 8) for _ in range(3)]
        vec = vector(arr)
        self.assertIsInstance(vec, ndarray)
        self.assertEqual(vec.dtype, float64)
        self.assertListEqual(vec.tolist(), arr)
        arr = randvec().tolist()
        vec = vector(arr)
        self.assertIsInstance(vec, ndarray)
        self.assertEqual(vec.dtype, float64)
        self.assertListEqual(vec.tolist(), arr)

    def test_canon(self):
        vec = randvec()
        for i in range(3):
            while vec[i] == 0:
                vec[i] = normalvariate(0, 1)
            vec[i] = abs(vec[i])
            self.assertListEqual(canon(vec).tolist(), vec.tolist())
            vec[i] = -vec[i]
            self.assertListEqual(canon(vec).tolist(), (-vec).tolist())
            vec[i] = 0
        self.assertListEqual(canon(vec).tolist(), vec.tolist())

    def test_normalize(self):
        vec, norm = randnonzerovec()
        self.assertListEqual(normalize(vec).tolist(), (vec / norm).tolist())

    def test_diff(self):
        vec1 = randvec()
        vec2 = vec1
        while (vec1 == vec2).all():
            vec2 = randvec()
        self.assertEqual(diff(vec1, vec1), 0)
        self.assertEqual(diff(vec2, vec2), 0)
        self.assertGreaterEqual(diff(vec1, vec2), abs(vec1 - vec2).max())

    def test_same(self):
        vec1 = randvec()
        vec2 = vec1
        while abs(vec1 - vec2).max() <= TOL:
            vec2 = randvec()
        self.assertTrue(same(vec1, vec1, 0))
        self.assertTrue(same(vec2, vec2, 0))
        self.assertFalse(same(vec1, vec2, TOL))
        self.assertTrue(same(vec1, vec1 + perturb(), TOL))
        self.assertTrue(same(vec2, vec2 + perturb(), TOL))
        self.assertFalse(same(vec1, vec1 + 2 * perturb(), TOL))
        self.assertFalse(same(vec2, vec2 + 2 * perturb(), TOL))

    def test_indep(self):
        vec = randvec()
        self.assertEqual(indep(vec, vec), 0)
        self.assertEqual(indep(vec, -vec), 0)
        self.assertEqual(indep(vec, 2 * vec), 0)
        self.assertEqual(indep(vec, 0 * vec), 0)
        self.assertGreater(indep(vec, vec + perturb()), 0)

    def test_parallel(self):
        vec = randvec()
        self.assertTrue(parallel(vec, vec, 0))
        self.assertTrue(parallel(vec, -vec, 0))
        self.assertTrue(parallel(vec, 2 * vec, 0))
        self.assertTrue(parallel(vec, 0 * vec, 0))
        self.assertTrue(parallel(vec, vec + perturb(), 4 * TOL))
        self.assertFalse(parallel(vec, vec + 2 * perturb(), TOL))

    def test_perpendicular(self):
        vec1 = randvec()
        vec2 = zeros(3)
        while (vec2 == 0).all():
            vec2 = randvec()
            vec2 -= vec2.dot(vec1) / vec1.dot(vec1) * vec1
        self.assertFalse(perpendicular(vec1, vec1, TOL))
        self.assertFalse(perpendicular(vec1, -vec1, TOL))
        self.assertFalse(perpendicular(vec1, 2 * vec1, TOL))
        self.assertTrue(perpendicular(vec1, 0 * vec1, TOL))
        self.assertTrue(perpendicular(vec1, vec2, TOL))
        self.assertTrue(perpendicular(vec1, vec2 + perturb(), 4 * TOL))

    def test_translate(self):
        vec = randvec()
        translation = randvec()
        self.assertListEqual(
            translate(vec, translation).tolist(), (vec + translation).tolist()
        )

    def test_invert(self):
        vec = randvec()
        self.assertListEqual(invert(vec).tolist(), (-vec).tolist())

    def test_move2(self):
        vec = randvec()
        normal = randunitvec()
        coef1, coef2 = [normalvariate(0, 1) for _ in range(2)]
        base = vec.dot(normal) * normal
        projection = vec - base
        perpendicular = cross(normal, projection)
        res = base + projection * coef1 + perpendicular * coef2
        self.assertListEqual(
            move2(vec, normal, coef1, coef2).tolist(), res.tolist()
        )

    def test_rotate(self):
        vec = randvec()
        rotation = randunitvec()
        angle = 2 * pi * random()
        x, y, z = rotation
        c = cos(angle)
        s = sin(angle)
        mat = array(
            [
                [
                    c + x * x * (1 - c),
                    x * y * (1 - c) - z * s,
                    x * z * (1 - c) + y * s,
                ],
                [
                    y * x * (1 - c) + z * s,
                    c + y * y * (1 - c),
                    y * z * (1 - c) - x * s,
                ],
                [
                    z * x * (1 - c) - y * s,
                    z * y * (1 - c) + x * s,
                    c + z * z * (1 - c),
                ],
            ]
        )
        self.assertLessEqual(
            abs(rotate(vec, rotation, angle) - mat @ vec).max(), TOL
        )

    def test_reflect(self):
        vec = randvec()
        reflection = randunitvec()
        x, y, z = reflection
        mat = array(
            [
                [1 - 2 * x * x, -2 * x * y, -2 * x * z],
                [-2 * x * y, 1 - 2 * y * y, -2 * y * z],
                [-2 * x * z, -2 * y * z, 1 - 2 * z * z],
            ]
        )
        self.assertLessEqual(
            abs(reflect(vec, reflection) - mat @ vec).max(), TOL
        )


if __name__ == "__main__":
    main()
