from .init import (
    TestCase,
    main,
    sin,
    cos,
    acos,
    randint,
    normalvariate,
    clip,
    empty,
    zeros,
    cross as cross_,
    norm as norm_,
    ndarray,
    float64,
    randangle,
    randvec,
    randunitvec,
    perturb,
    orthperturb,
)

from symmtools import (
    TOL,
    clamp,
    rational,
    vector,
    sqnorm,
    norm,
    cross,
    normalize,
    orthogonalize,
    canonicalize,
    diff,
    same,
    zero,
    indep,
    unitindep,
    parallel,
    unitparallel,
    perpendicular,
    orthvec,
    angle,
    intersectangle,
    translate,
    invert,
    trigrotate,
    rotate,
    reflect,
    trigrotmat,
    rotmat,
    reflmat,
    inertia,
    signvar,
    circshift,
)


class TestVecOp(TestCase):
    def test_clamp(self) -> None:
        low = normalvariate(0.0, 1.0)
        high = normalvariate(0.0, 1.0)
        if high < low:
            low, high = high, low
        val = normalvariate(0.0, 1.0)
        self.assertEqual(clamp(val, low, high), clip(val, low, high))

    def test_rational(self) -> None:
        nom1 = randint(-8, 8)
        denom1 = randint(1, 8)
        ratio1 = nom1 / denom1
        nom2, denom2 = rational(ratio1, TOL)
        ratio2 = nom2 / denom2
        self.assertAlmostEqual(ratio1, ratio2, delta=TOL)

    def test_vector(self) -> None:
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

    def test_sqnorm(self) -> None:
        vec = randvec()
        self.assertEqual(sqnorm(vec), vec.dot(vec))

    def test_norm(self) -> None:
        vec = randvec()
        self.assertEqual(norm(vec), norm_(vec))

    def test_cross(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        self.assertListEqual(
            cross(vec1, vec2).tolist(), cross_(vec1, vec2).tolist()
        )

    def test_normalize(self) -> None:
        vec = randvec(True)
        self.assertListEqual(
            normalize(vec).tolist(), (vec / norm_(vec)).tolist()
        )

    def test_orthogonalize(self) -> None:
        vec1 = randvec()
        vec2 = randunitvec()
        self.assertLessEqual(orthogonalize(vec1, vec2).dot(vec2), TOL)

    def test_canonicalize(self) -> None:
        vec = randvec()
        for i in range(3):
            while vec[i] == 0.0:
                vec[i] = normalvariate(0.0, 1.0)
            vec[i] = abs(vec[i])
            self.assertListEqual(canonicalize(vec).tolist(), vec.tolist())
            vec[i] = -vec[i]
            self.assertListEqual(canonicalize(vec).tolist(), (-vec).tolist())
            vec[i] = 0.0
        self.assertListEqual(canonicalize(vec).tolist(), vec.tolist())

    def test_diff(self) -> None:
        vec1 = randvec()
        vec2 = vec1
        while (vec1 == vec2).all():
            vec2 = randvec()
        self.assertEqual(diff(vec1, vec1), 0.0)
        self.assertEqual(diff(vec2, vec2), 0.0)
        self.assertGreaterEqual(diff(vec1, vec2), abs(vec1 - vec2).max())

    def test_same(self) -> None:
        vec1 = randvec()
        vec2 = vec1
        while diff(vec1, vec2) <= TOL:
            vec2 = randvec()
        self.assertTrue(same(vec1, vec1, 0.0))
        self.assertTrue(same(vec2, vec2, 0.0))
        self.assertFalse(same(vec1, vec2, TOL))
        self.assertTrue(same(vec1, vec1 + perturb(), TOL))
        self.assertTrue(same(vec2, vec2 + perturb(), TOL))
        self.assertFalse(same(vec1, vec1 + 2.0 * perturb(), TOL))
        self.assertFalse(same(vec2, vec2 + 2.0 * perturb(), TOL))

    def test_zero(self) -> None:
        self.assertTrue(zero(perturb(), TOL))

    def test_indep(self) -> None:
        vec = randvec()
        self.assertEqual(indep(vec, vec), 0.0)
        self.assertEqual(indep(vec, -vec), 0.0)
        self.assertEqual(indep(vec, 2.0 * vec), 0.0)
        self.assertEqual(indep(vec, 0.0 * vec), 0.0)
        self.assertGreater(indep(vec, vec + perturb()), 0.0)

    def test_unitindep(self) -> None:
        vec = randunitvec()
        self.assertEqual(unitindep(vec, vec), 0.0)
        self.assertEqual(unitindep(vec, -vec), 0.0)
        self.assertGreater(indep(vec, vec + perturb()), 0.0)

    def test_parallel(self) -> None:
        vec = randvec()
        self.assertTrue(parallel(vec, vec, 0.0))
        self.assertTrue(parallel(vec, -vec, 0.0))
        self.assertTrue(parallel(vec, 2.0 * vec, 0.0))
        self.assertTrue(parallel(vec, 0.0 * vec, 0.0))
        self.assertTrue(parallel(vec, vec + perturb(), 4.0 * TOL))
        self.assertFalse(
            parallel(vec, vec + 4.0 * orthperturb(normalize(vec)), TOL)
        )

    def test_unitparallel(self) -> None:
        vec1 = randunitvec()
        self.assertTrue(unitparallel(vec1, vec1, 0.0))
        self.assertTrue(unitparallel(vec1, -vec1, 0.0))
        vec2 = vec1 + perturb()
        vec2 /= norm_(vec2)
        self.assertTrue(unitparallel(vec1, vec2, 4.0 * TOL))
        vec2 = vec1 + 4.0 * orthperturb(vec1)
        vec2 /= norm_(vec2)
        self.assertFalse(unitparallel(vec1, vec2, TOL))

    def test_perpendicular(self) -> None:
        vec1 = randvec()
        vec2 = zeros(3)
        while (vec2 == 0.0).all():
            vec2 = randvec()
            vec2 -= vec2.dot(vec1) / vec1.dot(vec1) * vec1
        self.assertFalse(perpendicular(vec1, vec1, TOL))
        self.assertFalse(perpendicular(vec1, -vec1, TOL))
        self.assertFalse(perpendicular(vec1, 2.0 * vec1, TOL))
        self.assertTrue(perpendicular(vec1, 0.0 * vec1, TOL))
        self.assertTrue(perpendicular(vec1, vec2, TOL))
        self.assertTrue(perpendicular(vec1, vec2 + perturb(), 4.0 * TOL))

    def test_orthvec(self) -> None:
        vec = randunitvec()
        self.assertLessEqual(orthvec(vec).dot(vec), TOL)

    def test_angle(self) -> None:
        vec1 = randvec(True)
        vec2 = randvec(True)
        self.assertAlmostEqual(
            angle(vec1, vec2),
            acos(
                clip(vec1.dot(vec2) / (norm_(vec1) * norm_(vec2)), -1.0, 1.0)
            ),
            delta=TOL,
        )

    def test_intersectangle(self) -> None:
        vec1 = randvec(True)
        vec2 = randvec(True)
        if vec1.dot(vec2) < 0.0:
            vec2 = -vec2
        ang = acos(
            clip(vec1.dot(vec2) / (norm_(vec1) * norm_(vec2)), -1.0, 1.0)
        )
        self.assertAlmostEqual(intersectangle(vec1, vec2), ang, delta=TOL)
        self.assertAlmostEqual(intersectangle(vec1, -vec2), ang, delta=TOL)
        self.assertAlmostEqual(intersectangle(-vec1, vec2), ang, delta=TOL)
        self.assertAlmostEqual(intersectangle(-vec1, -vec2), ang, delta=TOL)

    def test_translate(self) -> None:
        vec = randvec()
        transl = randvec()
        self.assertListEqual(
            translate(vec, transl).tolist(), (vec + transl).tolist()
        )

    def test_invert(self) -> None:
        vec = randvec()
        self.assertListEqual(invert(vec).tolist(), (-vec).tolist())

    def test_trigrotate(self) -> None:
        vec = randvec()
        axis = randunitvec()
        ang = randangle()
        sin_ang = sin(ang)
        cos_ang = cos(ang)
        mat = trigrotmat(axis, cos_ang, sin_ang)
        self.assertLessEqual(
            diff(trigrotate(vec, axis, cos_ang, sin_ang), mat @ vec), TOL
        )

    def test_rotate(self) -> None:
        vec = randvec()
        axis = randunitvec()
        ang = randangle()
        mat = rotmat(axis, ang)
        self.assertLessEqual(diff(rotate(vec, axis, ang), mat @ vec), TOL)

    def test_reflect(self) -> None:
        vec = randvec()
        normal = randunitvec()
        mat = reflmat(normal)
        self.assertLessEqual(diff(reflect(vec, normal), mat @ vec), TOL)

    def test_inertia(self) -> None:
        vecs = [randvec() for _ in range(8)]
        mat = empty((3, 3))
        mat[0, 0] = sum(vec[1] ** 2 + vec[2] ** 2 for vec in vecs)
        mat[1, 1] = sum(vec[0] ** 2 + vec[2] ** 2 for vec in vecs)
        mat[2, 2] = sum(vec[0] ** 2 + vec[1] ** 2 for vec in vecs)
        mat[0, 1] = mat[1, 0] = -sum(vec[0] * vec[1] for vec in vecs)
        mat[1, 2] = mat[2, 1] = -sum(vec[1] * vec[2] for vec in vecs)
        mat[2, 0] = mat[0, 2] = -sum(vec[2] * vec[0] for vec in vecs)
        self.assertLessEqual(abs(inertia(vecs) - mat).max(), TOL)

    def test_signvar(self) -> None:
        vec = zeros(3)
        for i in range(len(vec)):
            while vec[i] == 0.0:
                vec[i] = normalvariate(0.0, 1.0)
        x, y, z = vec
        vecs = set(tuple(elem) for elem in signvar(vec))
        self.assertSetEqual(
            vecs,
            {
                (x, y, z),
                (-x, y, z),
                (x, -y, z),
                (x, y, -z),
                (-x, -y, z),
                (-x, y, -z),
                (x, -y, -z),
                (-x, -y, -z),
            },
        )
        vecs = set(tuple(elem) for elem in signvar(vec, 0, True))
        self.assertSetEqual(
            vecs, {(x, y, z), (-x, y, z), (x, -y, z), (-x, -y, z)}
        )
        vecs = set(tuple(elem) for elem in signvar(vec, 1))
        self.assertSetEqual(
            vecs, {(x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z)}
        )
        vecs = set(tuple(elem) for elem in signvar(vec, -1))
        self.assertSetEqual(
            vecs, {(-x, y, z), (x, -y, z), (x, y, -z), (-x, -y, -z)}
        )
        vec = zeros(2)
        vecs = set(tuple(elem) for elem in signvar(vec))
        self.assertSetEqual(vecs, {(0.0, 0.0)})
        for i in range(len(vec)):
            while vec[i] == 0.0:
                vec[i] = normalvariate(0.0, 1.0)
        x, y = vec
        vec[0] = 0.0
        vecs = set(tuple(elem) for elem in signvar(vec))
        self.assertSetEqual(vecs, {(0.0, y), (0.0, -y)})
        vec[0] = x
        vec[1] = 0.0
        vecs = set(tuple(elem) for elem in signvar(vec))
        self.assertSetEqual(vecs, {(x, 0.0), (-x, 0.0)})
        vec[1] = y
        vecs = set(tuple(elem) for elem in signvar(vec))
        self.assertSetEqual(vecs, {(x, y), (-x, y), (x, -y), (-x, -y)})

    def test_circshift(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        x1, y1, z1 = vec1
        x2, y2, z2 = vec2
        vecs = set(tuple(elem) for elem in circshift((vec1, vec2)))
        self.assertSetEqual(
            vecs,
            {
                (x1, y1, z1),
                (y1, z1, x1),
                (z1, x1, y1),
                (x2, y2, z2),
                (y2, z2, x2),
                (z2, x2, y2),
            },
        )


if __name__ == "__main__":
    main()
