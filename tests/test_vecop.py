from unittest import TestCase, main

from .tools import (
    randint,
    normalvariate,
    zeros,
    cross,
    norm,
    ndarray,
    float64,
    randvec,
    randunitvec,
    randne0vec,
    perturb,
    randangle,
)

from symmtools import (
    TOL,
    vector,
    canonicalize,
    normalize,
    orthogonalize,
    diff,
    same,
    indep,
    indepunit,
    parallel,
    perpendicular,
    translate,
    invert,
    move2,
    rotate,
    reflect,
    rotmat,
    reflmat,
)


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

    def test_canonicalize(self):
        vec = randvec()
        for i in range(3):
            while vec[i] == 0:
                vec[i] = normalvariate(0, 1)
            vec[i] = abs(vec[i])
            self.assertListEqual(canonicalize(vec).tolist(), vec.tolist())
            vec[i] = -vec[i]
            self.assertListEqual(canonicalize(vec).tolist(), (-vec).tolist())
            vec[i] = 0
        self.assertListEqual(canonicalize(vec).tolist(), vec.tolist())

    def test_normalize(self):
        vec = randne0vec()
        self.assertListEqual(
            normalize(vec).tolist(), (vec / norm(vec)).tolist()
        )

    def test_orthogonalize(self):
        vec1 = randvec()
        vec2 = randunitvec()
        self.assertLessEqual(orthogonalize(vec1, vec2).dot(vec2), TOL)

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
        while diff(vec1, vec2) <= TOL:
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

    def test_indepunit(self):
        vec = randunitvec()
        self.assertEqual(indepunit(vec, vec), 0)
        self.assertEqual(indepunit(vec, -vec), 0)
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
        angle = randangle()
        mat = rotmat(rotation, angle)
        self.assertLessEqual(
            diff(rotate(vec, rotation, angle), mat @ vec), TOL
        )

    def test_reflect(self):
        vec = randvec()
        reflection = randunitvec()
        mat = reflmat(reflection)
        self.assertLessEqual(diff(reflect(vec, reflection), mat @ vec), TOL)


if __name__ == "__main__":
    main()
