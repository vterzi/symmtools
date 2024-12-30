from .init import (
    TestCase,
    main,
    sin,
    cos,
    acos,
    clip,
    array,
    empty,
    zeros,
    cross as cross_,
    norm as norm_,
    TOL,
    vec3D,
    randfloat,
    randangle,
    randvec,
    randunitvec,
    randmat,
    perturb,
    orthperturb,
)

from symmtools.linalg3d import (
    vector,
    matrix,
    pos,
    neg,
    add,
    sub,
    mul,
    lincomb2,
    lincomb3,
    dot,
    sqnorm,
    norm,
    cross,
    normalize,
    orthogonalize,
    canonicalize,
    zero,
    diff,
    same,
    indep,
    unitindep,
    parallel,
    unitparallel,
    perpendicular,
    orthvec,
    angle,
    transpose,
    matmulvec,
    vecmulmat,
    matmulmat,
    intersectangle,
    trigrotate,
    rotate,
    reflect,
    trigrotmat,
    rotmat,
    reflmat,
    inertia,
    symmeig,
)


class TestLinAlg3D(TestCase):
    def test_vector(self) -> None:
        arr = array(randvec()).tolist()
        vec = vector(arr)
        self.assertTupleEqual(vec, vec3D(arr))
        self.assertRaises(IndexError, vector, arr[:2])

    def test_matrix(self) -> None:
        arr = array(randmat()).tolist()
        mat = matrix(arr)
        self.assertLessEqual(abs(mat - array(arr)).max(), TOL)
        self.assertRaises(IndexError, matrix, arr[:2])

    def test_pos(self) -> None:
        vec = randvec()
        self.assertTupleEqual(pos(vec), vec3D(+array(vec)))

    def test_neg(self) -> None:
        vec = randvec()
        self.assertTupleEqual(neg(vec), vec3D(-array(vec)))

    def test_add(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        self.assertTupleEqual(add(vec1, vec2), vec3D(array(vec1) + vec2))

    def test_sub(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        self.assertTupleEqual(sub(vec1, vec2), vec3D(array(vec1) - vec2))

    def test_mul(self) -> None:
        vec = randvec()
        scalar = randfloat()
        self.assertTupleEqual(mul(vec, scalar), vec3D(array(vec) * scalar))

    def test_lincomb2(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        scalar1 = randfloat()
        scalar2 = randfloat()
        self.assertTupleEqual(
            lincomb2(vec1, scalar1, vec2, scalar2),
            vec3D(array(vec1) * scalar1 + array(vec2) * scalar2),
        )

    def test_lincomb3(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        vec3 = randvec()
        scalar1 = randfloat()
        scalar2 = randfloat()
        scalar3 = randfloat()
        self.assertTupleEqual(
            lincomb3(vec1, scalar1, vec2, scalar2, vec3, scalar3),
            vec3D(
                array(vec1) * scalar1
                + array(vec2) * scalar2
                + array(vec3) * scalar3
            ),
        )

    def test_dot(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        self.assertEqual(dot(vec1, vec2), array(vec1).dot(vec2))

    def test_sqnorm(self) -> None:
        vec = randvec()
        self.assertEqual(sqnorm(vec), array(vec).dot(vec))

    def test_norm(self) -> None:
        vec = randvec()
        self.assertAlmostEqual(norm(vec), norm_(vec), delta=TOL)

    def test_cross(self) -> None:
        vec1 = randvec()
        vec2 = randvec()
        self.assertTupleEqual(cross(vec1, vec2), vec3D(cross_(vec1, vec2)))

    def test_normalize(self) -> None:
        vec = randvec(True)
        self.assertLessEqual(
            abs(normalize(vec) - array(vec) / norm_(vec)).max(), TOL
        )

    def test_orthogonalize(self) -> None:
        vec1 = randvec()
        vec2 = randunitvec()
        self.assertLessEqual(array(vec2).dot(orthogonalize(vec1, vec2)), TOL)

    def test_canonicalize(self) -> None:
        arr = array(randvec())
        for i in range(3):
            while arr[i] == 0.0:
                arr[i] = randfloat()
            arr[i] = abs(arr[i])
            self.assertTupleEqual(canonicalize(vec3D(arr)), vec3D(arr))
            arr[i] = -arr[i]
            self.assertTupleEqual(canonicalize(vec3D(arr)), vec3D(-arr))
            arr[i] = 0.0
        self.assertTupleEqual(canonicalize(vec3D(arr)), vec3D(arr))

    def test_diff(self) -> None:
        vec1 = randvec()
        vec2 = vec1
        while vec1 == vec2:
            vec2 = randvec()
        self.assertEqual(diff(vec1, vec1), 0.0)
        self.assertEqual(diff(vec2, vec2), 0.0)
        self.assertGreaterEqual(
            diff(vec1, vec2), abs(array(vec1) - vec2).max()
        )

    def test_same(self) -> None:
        vec1 = randvec()
        vec2 = vec1
        while abs(array(vec1) - vec2).max() <= TOL:
            vec2 = randvec()
        self.assertTrue(same(vec1, vec1, 0.0))
        self.assertTrue(same(vec2, vec2, 0.0))
        self.assertFalse(same(vec1, vec2, TOL))
        self.assertTrue(same(vec1, vec3D(array(perturb()) + vec1), TOL))
        self.assertTrue(same(vec2, vec3D(array(perturb()) + vec2), TOL))
        self.assertFalse(same(vec1, vec3D(2.0 * array(perturb()) + vec1), TOL))
        self.assertFalse(same(vec2, vec3D(2.0 * array(perturb()) + vec2), TOL))

    def test_zero(self) -> None:
        self.assertTrue(zero(perturb(), TOL))

    def test_indep(self) -> None:
        vec = randvec()
        arr = array(vec)
        self.assertEqual(indep(vec, vec3D(arr)), 0.0)
        self.assertEqual(indep(vec, vec3D(-arr)), 0.0)
        self.assertEqual(indep(vec, vec3D(2.0 * arr)), 0.0)
        self.assertEqual(indep(vec, vec3D(0.0 * arr)), 0.0)
        self.assertGreater(indep(vec, vec3D(array(perturb()) + vec)), 0.0)

    def test_unitindep(self) -> None:
        vec = randunitvec()
        arr = array(vec)
        self.assertEqual(unitindep(vec, vec3D(arr)), 0.0)
        self.assertEqual(unitindep(vec, vec3D(-arr)), 0.0)
        self.assertGreater(indep(vec, vec3D(array(perturb()) + vec)), 0.0)

    def test_parallel(self) -> None:
        vec = randvec()
        arr = array(vec)
        self.assertTrue(parallel(vec, vec3D(arr), 0.0))
        self.assertTrue(parallel(vec, vec3D(-arr), 0.0))
        self.assertTrue(parallel(vec, vec3D(2.0 * arr), 0.0))
        self.assertTrue(parallel(vec, vec3D(0.0 * arr), 0.0))
        self.assertTrue(
            parallel(vec, vec3D(array(perturb()) + vec), 4.0 * TOL)
        )
        self.assertFalse(
            parallel(
                vec,
                vec3D(4.0 * array(orthperturb(vec3D(arr / norm_(vec)))) + vec),
                TOL,
            )
        )

    def test_unitparallel(self) -> None:
        vec = randunitvec()
        arr = array(vec)
        self.assertTrue(unitparallel(vec, vec3D(arr), 0.0))
        self.assertTrue(unitparallel(vec, vec3D(-arr), 0.0))
        arr = array(perturb()) + vec
        arr /= norm_(arr)
        self.assertTrue(unitparallel(vec, vec3D(arr), 4.0 * TOL))
        arr = 4.0 * array(orthperturb(vec)) + vec
        arr /= norm_(arr)
        self.assertFalse(unitparallel(vec, vec3D(arr), TOL))

    def test_perpendicular(self) -> None:
        vec = randvec()
        arr1 = array(vec)
        arr2 = zeros(3)
        while (arr2 == 0.0).all():
            arr2 = array(randvec())
            arr2 -= arr2.dot(arr1) / arr1.dot(arr1) * arr1
        self.assertFalse(perpendicular(vec, vec3D(arr1), TOL))
        self.assertFalse(perpendicular(vec, vec3D(-arr1), TOL))
        self.assertFalse(perpendicular(vec, vec3D(2.0 * arr1), TOL))
        self.assertTrue(perpendicular(vec, vec3D(0.0 * arr1), TOL))
        self.assertTrue(perpendicular(vec, vec3D(arr2), TOL))
        self.assertTrue(perpendicular(vec, vec3D(arr2 + perturb()), 4.0 * TOL))

    def test_orthvec(self) -> None:
        vec = randunitvec()
        self.assertLessEqual(array(vec).dot(orthvec(vec)), TOL)

    def test_angle(self) -> None:
        vec1 = randvec(True)
        vec2 = randvec(True)
        self.assertAlmostEqual(
            angle(vec1, vec2),
            acos(
                clip(
                    array(vec1).dot(vec2) / (norm_(vec1) * norm_(vec2)),
                    -1.0,
                    1.0,
                )
            ),
            delta=TOL,
        )

    def test_intersectangle(self) -> None:
        arr1 = array(randvec(True))
        arr2 = array(randvec(True))
        if arr1.dot(arr2) < 0.0:
            arr2 = -arr2
        ang = acos(
            clip(arr1.dot(arr2) / (norm_(arr1) * norm_(arr2)), -1.0, 1.0)
        )
        self.assertAlmostEqual(
            intersectangle(vec3D(arr1), vec3D(arr2)), ang, delta=TOL
        )
        self.assertAlmostEqual(
            intersectangle(vec3D(arr1), vec3D(-arr2)), ang, delta=TOL
        )
        self.assertAlmostEqual(
            intersectangle(vec3D(-arr1), vec3D(arr2)), ang, delta=TOL
        )
        self.assertAlmostEqual(
            intersectangle(vec3D(-arr1), vec3D(-arr2)), ang, delta=TOL
        )

    def test_transpose(self) -> None:
        mat = randmat()
        self.assertLessEqual(abs(transpose(mat) - array(mat).T).max(), TOL)

    def test_matmulvec(self) -> None:
        mat = randmat()
        vec = randvec()
        self.assertTupleEqual(matmulvec(mat, vec), vec3D(array(mat) @ vec))

    def test_vecmulmat(self) -> None:
        vec = randvec()
        mat = randmat()
        self.assertLessEqual(
            abs(vecmulmat(vec, mat) - array(vec) @ mat).max(), TOL
        )

    def test_matmulmat(self) -> None:
        mat1 = randmat()
        mat2 = randmat()
        self.assertLessEqual(
            abs(matmulmat(mat1, mat2) - array(mat1) @ mat2).max(), TOL
        )

    def test_trigrotate(self) -> None:
        vec = randvec()
        axis = randunitvec()
        ang = randangle()
        sin_ang = sin(ang)
        cos_ang = cos(ang)
        mat = array(trigrotmat(axis, cos_ang, sin_ang))
        self.assertLessEqual(
            abs(trigrotate(vec, axis, cos_ang, sin_ang) - mat @ vec).max(),
            TOL,
        )

    def test_rotate(self) -> None:
        vec = randvec()
        axis = randunitvec()
        ang = randangle()
        mat = array(rotmat(axis, ang))
        self.assertLessEqual(
            abs(rotate(vec, axis, ang) - mat @ vec).max(), TOL
        )

    def test_reflect(self) -> None:
        vec = randvec()
        normal = randunitvec()
        mat = array(reflmat(normal))
        self.assertLessEqual(abs(reflect(vec, normal) - mat @ vec).max(), TOL)

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

    def test_symmeig(self) -> None:
        mat = randmat()
        eigvals1, eigvecs1 = symmeig(mat, fast=False)
        eigvals2, eigvecs2 = symmeig(mat)
        self.assertLessEqual(abs(array(eigvals1) - eigvals2).max(), TOL)
        self.assertLessEqual(
            float(norm_(cross_(eigvecs1[0], eigvecs2[0]))), TOL
        )
        self.assertLessEqual(
            float(norm_(cross_(eigvecs1[1], eigvecs2[1]))), TOL
        )
        self.assertLessEqual(
            float(norm_(cross_(eigvecs1[2], eigvecs2[2]))), TOL
        )


if __name__ == "__main__":
    main()
