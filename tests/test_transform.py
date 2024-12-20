from .init import (
    TestCase,
    main,
    array,
    eye,
    norm,
    randsign,
    randangle,
    randvec,
    randunitvec,
    perturb,
    orthperturb,
    TOL,
)

from symmtools.const import INF, TAU
from symmtools.linalg3d import (
    neg,
    add,
    mul,
    matmul,
    matprod,
    normalize,
    parallel,
    rotate,
    reflect,
    rotmat,
    reflmat,
)
from symmtools.primitive import Point
from symmtools.transform import (
    Transformation,
    Identity,
    Translation,
    Inversion,
    Rotation,
    Reflection,
    Rotoreflection,
)


class TestIdentity(TestCase):
    def test_call(self) -> None:
        transform = Identity()
        point = Point(randvec())
        self.assertEqual(transform(point), point)

    def test_representation(self) -> None:
        transform = Identity()
        string = "Identity()"
        self.assertEqual(str(transform), string)
        self.assertEqual(repr(transform), string)

    def test_comparison(self) -> None:
        transform = Identity()
        point = Point(randvec())
        self.assertEqual(transform.diff(transform), 0.0)
        self.assertEqual(transform.diff(point), INF)
        self.assertTrue(transform.same(transform, 0.0))
        self.assertFalse(transform.same(point, TOL))
        self.assertEqual(transform, transform)
        self.assertNotEqual(transform, point)

    def test_mat(self) -> None:
        transform = Identity()
        self.assertTrue((array(transform.mat) == eye(3)).all())

    def test_transformation(self) -> None:
        transform_: Transformation
        transform = Identity()
        self.assertEqual(transform.copy(), transform)
        transform_ = Identity()
        self.assertEqual(transform_(transform), transform)
        transform_ = Translation(randvec())
        self.assertEqual(transform_(transform), transform)
        transform_ = Inversion()
        self.assertEqual(transform_(transform), transform)
        transform_ = Rotation(randunitvec(), randangle(True))
        self.assertEqual(transform_(transform), transform)
        transform_ = Reflection(randunitvec())
        self.assertEqual(transform_(transform), transform)
        transform_ = Rotoreflection(randunitvec(), randangle(True))
        self.assertEqual(transform_(transform), transform)


class TestTranslation(TestCase):
    def test_init(self) -> None:
        vec = randvec()
        transform = Translation(vec)
        self.assertTupleEqual(transform.vec, vec)

    def test_call(self) -> None:
        vec = randvec()
        transform = Translation(vec)
        pos = randvec()
        point = Point(pos)
        self.assertEqual(transform(point), Point(add(pos, vec)))

    def test_representation(self) -> None:
        vec = randvec()
        transform = Translation(vec)
        string = f"Translation({vec})"
        string = string.replace(" ", "")
        self.assertEqual(str(transform), string)
        self.assertEqual(repr(transform), string)

    def test_comparison(self) -> None:
        vec = randvec()
        transform = Translation(vec)
        point = Point(randvec())
        self.assertEqual(transform.diff(transform), 0.0)
        self.assertEqual(transform.diff(point), INF)
        self.assertEqual(transform.diff(Translation(add(vec, perturb()))), TOL)
        self.assertTrue(transform.same(transform, 0.0))
        self.assertFalse(transform.same(point, TOL))
        self.assertTrue(transform.same(Translation(add(vec, perturb())), TOL))
        self.assertFalse(
            transform.same(Translation(add(vec, mul(perturb(), 2.0))), TOL)
        )
        self.assertEqual(transform, transform)
        self.assertNotEqual(transform, point)

    def test_mat(self) -> None:
        vec = randvec()
        transform = Translation(vec)
        self.assertRaises(NotImplementedError, lambda: transform.mat)

    def test_transformation(self) -> None:
        transform_: Transformation
        vec = randvec()
        transform = Translation(vec)
        self.assertEqual(transform.copy(), transform)
        transform_ = Identity()
        self.assertEqual(transform_(transform), transform)
        vec_ = randvec()
        transform_ = Translation(vec_)
        self.assertEqual(transform_(transform), Translation(add(vec, vec_)))
        transform_ = Inversion()
        self.assertEqual(transform_(transform), Translation(neg(vec)))
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotation(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Translation(rotate(vec, vec_, angle_)), TOL
            )
        )
        vec_ = randunitvec()
        transform_ = Reflection(vec_)
        self.assertTrue(
            transform_(transform).same(Translation(reflect(vec, vec_)), TOL)
        )
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotoreflection(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Translation(reflect(rotate(vec, vec_, angle_), vec_)), TOL
            )
        )


class TestInversion(TestCase):
    def test_call(self) -> None:
        transform = Inversion()
        pos = randvec()
        point = Point(pos)
        self.assertEqual(transform(point), Point(neg(pos)))

    def test_representation(self) -> None:
        transform = Inversion()
        string = "Inversion()"
        self.assertEqual(str(transform), string)
        self.assertEqual(repr(transform), string)

    def test_comparison(self) -> None:
        transform = Inversion()
        point = Point(randvec())
        self.assertEqual(transform.diff(transform), 0.0)
        self.assertEqual(transform.diff(point), INF)
        self.assertTrue(transform.same(transform, 0.0))
        self.assertFalse(transform.same(point, TOL))
        self.assertEqual(transform, transform)
        self.assertNotEqual(transform, point)

    def test_mat(self) -> None:
        transform = Inversion()
        mat = -eye(3)
        self.assertTrue((array(transform.mat) == mat).all())

    def test_transformation(self) -> None:
        transform_: Transformation
        transform = Inversion()
        self.assertEqual(transform.copy(), transform)
        transform_ = Identity()
        self.assertEqual(transform_(transform), transform)
        transform_ = Translation(randvec())
        self.assertEqual(transform_(transform), transform)
        transform_ = Inversion()
        self.assertEqual(transform_(transform), transform)
        transform_ = Rotation(randunitvec(), randangle(True))
        self.assertEqual(transform_(transform), transform)
        transform_ = Reflection(randunitvec())
        self.assertEqual(transform_(transform), transform)
        transform_ = Rotoreflection(randunitvec(), randangle(True))
        self.assertEqual(transform_(transform), transform)


class TestRotation(TestCase):
    def test_init(self) -> None:
        vec = randvec(True)
        angle = randangle(True)
        transform = Rotation(vec, angle)
        self.assertTrue(parallel(transform.vec, vec, TOL))
        self.assertAlmostEqual(float(norm(transform.vec)), 1.0, delta=TOL)
        self.assertEqual(transform.angle, angle)
        self.assertAlmostEqual(
            Rotation(vec, angle + TAU).angle, angle, delta=TOL
        )
        self.assertAlmostEqual(
            Rotation(vec, angle - TAU).angle, angle, delta=TOL
        )
        self.assertRaises(ValueError, Rotation, (0.0, 0.0, 0.0), angle)
        self.assertRaises(ValueError, Rotation, vec, 0.0)
        self.assertRaises(ValueError, Rotation, vec, TAU)

    def test_call(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotation(vec, angle)
        pos = randvec()
        point = Point(pos)
        mat = rotmat(vec, angle)
        self.assertTrue(transform(point).same(Point(matmul(mat, pos)), TOL))

    def test_representation(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotation(vec, angle)
        string = f"Rotation({normalize(vec)},{angle})"
        string = string.replace(" ", "")
        self.assertEqual(str(transform), string)
        self.assertEqual(repr(transform), string)

    def test_comparison(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotation(vec, angle)
        point = Point(randvec())
        self.assertEqual(transform.diff(transform), 0.0)
        self.assertEqual(transform.diff(point), INF)
        self.assertEqual(
            transform.diff(Rotation(vec, angle + randsign() * TOL)), TOL
        )
        self.assertTrue(transform.same(transform, 0.0))
        self.assertFalse(transform.same(point, TOL))
        self.assertTrue(transform.same(Rotation(vec, angle + TAU), TOL))
        self.assertTrue(transform.same(Rotation(vec, angle - TAU), TOL))
        self.assertTrue(transform.same(Rotation(neg(vec), TAU - angle), TOL))
        self.assertTrue(
            Rotation(vec, 0.5 * TOL).same(Rotation(vec, TAU - 0.5 * TOL), TOL)
        )
        self.assertTrue(
            transform.same(Rotation(add(vec, orthperturb(vec)), angle), TOL)
        )
        self.assertFalse(
            transform.same(
                Rotation(add(vec, mul(orthperturb(vec), 2.0)), angle), TOL
            )
        )
        self.assertEqual(transform, transform)
        self.assertNotEqual(transform, point)

    def test_mat(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotation(vec, angle)
        mat = rotmat(vec, angle)
        self.assertLessEqual(abs(array(transform.mat) - mat).max(), TOL)

    def test_transformation(self) -> None:
        transform_: Transformation
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotation(vec, angle)
        self.assertEqual(transform.copy(), transform)
        transform_ = Identity()
        self.assertEqual(transform_(transform), transform)
        vec_ = randvec()
        transform_ = Translation(vec_)
        self.assertEqual(transform_(transform), transform)
        transform_ = Inversion()
        self.assertEqual(transform_(transform), Rotation(neg(vec), angle))
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotation(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Rotation(rotate(vec, vec_, angle_), angle), TOL
            )
        )
        vec_ = randunitvec()
        transform_ = Reflection(vec_)
        self.assertTrue(
            transform_(transform).same(
                Rotation(reflect(vec, vec_), angle), TOL
            )
        )
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotoreflection(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Rotation(reflect(rotate(vec, vec_, angle_), vec_), angle), TOL
            )
        )


class TestReflection(TestCase):
    def test_init(self) -> None:
        vec = randvec(True)
        transform = Reflection(vec)
        self.assertTrue(parallel(transform.vec, vec, TOL))
        self.assertAlmostEqual(float(norm(transform.vec)), 1.0, delta=TOL)
        self.assertRaises(ValueError, Reflection, (0.0, 0.0, 0.0))

    def test_call(self) -> None:
        vec = randunitvec()
        transform = Reflection(vec)
        pos = randvec()
        point = Point(pos)
        mat = reflmat(vec)
        self.assertTrue(transform(point).same(Point(matmul(mat, pos)), TOL))

    def test_representation(self) -> None:
        vec = randunitvec()
        transform = Reflection(vec)
        string = f"Reflection({normalize(vec)})"
        string = string.replace(" ", "")
        self.assertEqual(str(transform), string)
        self.assertEqual(repr(transform), string)

    def test_comparison(self) -> None:
        vec = randunitvec()
        transform = Reflection(vec)
        point = Point(randvec())
        self.assertEqual(transform.diff(transform), 0.0)
        self.assertEqual(transform.diff(point), INF)
        self.assertTrue(transform.same(transform, 0.0))
        self.assertFalse(transform.same(point, TOL))
        self.assertTrue(transform.same(Reflection(neg(vec)), TOL))
        self.assertTrue(
            transform.same(Reflection(add(vec, orthperturb(vec))), TOL)
        )
        self.assertFalse(
            transform.same(
                Reflection(add(vec, mul(orthperturb(vec), 2.0))), TOL
            )
        )
        self.assertEqual(transform, transform)
        self.assertNotEqual(transform, point)

    def test_mat(self) -> None:
        vec = randunitvec()
        transform = Reflection(vec)
        mat = reflmat(vec)
        self.assertLessEqual(abs(array(transform.mat) - mat).max(), TOL)

    def test_transformation(self) -> None:
        transform_: Transformation
        vec = randunitvec()
        transform = Reflection(vec)
        self.assertEqual(transform.copy(), transform)
        transform_ = Identity()
        self.assertEqual(transform_(transform), transform)
        vec_ = randvec()
        transform_ = Translation(vec_)
        self.assertEqual(transform_(transform), transform)
        transform_ = Inversion()
        self.assertEqual(transform_(transform), transform)
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotation(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Reflection(rotate(vec, vec_, angle_)), TOL
            )
        )
        vec_ = randunitvec()
        transform_ = Reflection(vec_)
        self.assertTrue(
            transform_(transform).same(Reflection(reflect(vec, vec_)), TOL)
        )
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotoreflection(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Reflection(reflect(rotate(vec, vec_, angle_), vec_)), TOL
            )
        )


class TestRotoreflection(TestCase):
    def test_init(self) -> None:
        vec = randvec(True)
        angle = randangle(True)
        transform = Rotoreflection(vec, angle)
        self.assertTrue(parallel(transform.vec, vec, TOL))
        self.assertAlmostEqual(float(norm(transform.vec)), 1.0, delta=TOL)
        self.assertEqual(transform.angle, angle)
        self.assertAlmostEqual(
            Rotoreflection(vec, angle + TAU).angle, angle, delta=TOL
        )
        self.assertAlmostEqual(
            Rotoreflection(vec, angle - TAU).angle, angle, delta=TOL
        )
        self.assertRaises(ValueError, Rotoreflection, (0.0, 0.0, 0.0), angle)
        self.assertRaises(ValueError, Rotoreflection, vec, 0.0)
        self.assertRaises(ValueError, Rotoreflection, vec, TAU)

    def test_call(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotoreflection(vec, angle)
        pos = randvec()
        point = Point(pos)
        mat = matprod(reflmat(vec), rotmat(vec, angle))
        self.assertTrue(transform(point).same(Point(matmul(mat, pos)), TOL))

    def test_representation(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotoreflection(vec, angle)
        string = f"Rotoreflection({normalize(vec)},{angle})"
        string = string.replace(" ", "")
        self.assertEqual(str(transform), string)
        self.assertEqual(repr(transform), string)

    def test_comparison(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotoreflection(vec, angle)
        point = Point(randvec())
        self.assertEqual(transform.diff(transform), 0.0)
        self.assertEqual(transform.diff(point), INF)
        self.assertEqual(
            transform.diff(Rotoreflection(vec, angle + randsign() * TOL)), TOL
        )
        self.assertTrue(transform.same(transform, 0.0))
        self.assertFalse(transform.same(point, TOL))
        self.assertTrue(transform.same(Rotoreflection(vec, angle + TAU), TOL))
        self.assertTrue(transform.same(Rotoreflection(vec, angle - TAU), TOL))
        self.assertTrue(
            transform.same(Rotoreflection(neg(vec), TAU - angle), TOL)
        )
        self.assertTrue(
            Rotoreflection(vec, 0.5 * TOL).same(
                Rotoreflection(vec, TAU - 0.5 * TOL), TOL
            )
        )
        self.assertTrue(
            transform.same(
                Rotoreflection(add(vec, orthperturb(vec)), angle), TOL
            )
        )
        self.assertFalse(
            transform.same(
                Rotoreflection(add(vec, mul(orthperturb(vec), 2.0)), angle),
                TOL,
            )
        )
        self.assertEqual(transform, transform)
        self.assertNotEqual(transform, point)

    def test_mat(self) -> None:
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotoreflection(vec, angle)
        mat = matprod(reflmat(vec), rotmat(vec, angle))
        self.assertLessEqual(abs(array(transform.mat) - mat).max(), TOL)

    def test_transformation(self) -> None:
        transform_: Transformation
        vec = randunitvec()
        angle = randangle(True)
        transform = Rotoreflection(vec, angle)
        self.assertEqual(transform.copy(), transform)
        transform_ = Identity()
        self.assertEqual(transform_(transform), transform)
        vec_ = randvec()
        transform_ = Translation(vec_)
        self.assertEqual(transform_(transform), transform)
        transform_ = Inversion()
        self.assertEqual(
            transform_(transform), Rotoreflection(neg(vec), angle)
        )
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotation(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Rotoreflection(rotate(vec, vec_, angle_), angle), TOL
            )
        )
        vec_ = randunitvec()
        transform_ = Reflection(vec_)
        self.assertTrue(
            transform_(transform).same(
                Rotoreflection(reflect(vec, vec_), angle), TOL
            )
        )
        vec_ = randunitvec()
        angle_ = randangle(True)
        transform_ = Rotoreflection(vec_, angle_)
        self.assertTrue(
            transform_(transform).same(
                Rotoreflection(
                    reflect(rotate(vec, vec_, angle_), vec_), angle
                ),
                TOL,
            )
        )


if __name__ == "__main__":
    main()
