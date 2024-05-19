from .init import (
    TestCase,
    main,
    randvec,
    List,
)

from symmtools import (
    TAU,
    TOL,
    IdentityElem,
    InversionCenter,
    RotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    Transformable,
    Transformation,
    Identity,
    Inversion,
    Rotation,
    Reflection,
    Rotoreflection,
    Point,
    Elems,
    normalize,
)


class TestIdentityElem(TestCase):
    def test_transformations(self):
        symmelem = IdentityElem()
        self.assertSequenceEqual(symmelem.transformations(), [Identity()])

    def test_symb(self):
        symmelem = IdentityElem()
        self.assertEqual(symmelem.symb(), "E")

    def test_symmetric(self):
        symmelem = IdentityElem()
        points = [Point(randvec()) for _ in range(3)]
        self.assertTrue(symmelem.symmetric(Elems(points), TOL))


class TestInversionCenter(TestCase):
    def test_transformations(self):
        symmelem = InversionCenter()
        self.assertSequenceEqual(symmelem.transformations(), [Inversion()])

    def test_symb(self):
        symmelem = InversionCenter()
        self.assertEqual(symmelem.symb(), "i")

    def test_symmetric(self):
        symmelem = InversionCenter()
        points: List[Transformable] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in symmelem.transformations():
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Elems(points), TOL))


class TestRotationAxis(TestCase):
    def test_init(self):
        self.assertRaises(ValueError, RotationAxis, randvec(), 0)
        self.assertRaises(ValueError, RotationAxis, randvec(), -1)

    def test_transformations(self):
        vec = randvec()
        symmelem = RotationAxis(vec, 3)
        vec = normalize(vec)
        self.assertSequenceEqual(
            symmelem.transformations(),
            [Rotation(vec, 1 / 3 * TAU), Rotation(vec, 2 / 3 * TAU)],
        )
        symmelem = RotationAxis(vec, 4)
        vec = normalize(vec)
        self.assertSequenceEqual(
            symmelem.transformations(),
            [
                Rotation(vec, 1 / 4 * TAU),
                Rotation(vec, 2 / 4 * TAU),
                Rotation(vec, 3 / 4 * TAU),
            ],
        )

    def test_symb(self):
        order = 3
        symmelem = RotationAxis(randvec(), order)
        self.assertEqual(symmelem.symb(), f"C{order}")

    def test_symmetric(self):
        symmelem = RotationAxis(randvec(), 3)
        points: List[Transformable] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in symmelem.transformations():
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Elems(points), TOL))


class TestReflectionPlane(TestCase):
    def test_transformations(self):
        vec = randvec()
        symmelem = ReflectionPlane(vec)
        vec = normalize(vec)
        self.assertSequenceEqual(symmelem.transformations(), [Reflection(vec)])

    def test_symb(self):
        symmelem = ReflectionPlane(randvec())
        self.assertEqual(symmelem.symb(), "s")

    def test_symmetric(self):
        symmelem = ReflectionPlane(randvec())
        points: List[Transformable] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in symmelem.transformations():
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Elems(points), TOL))


class TestRotoreflectionAxis(TestCase):
    def test_init(self):
        self.assertRaises(ValueError, RotoreflectionAxis, randvec(), 0)
        self.assertRaises(ValueError, RotoreflectionAxis, randvec(), -1)

    def test_transformations(self):
        vec = randvec()
        symmelem = RotoreflectionAxis(vec, 3)
        vec = normalize(vec)
        self.assertSequenceEqual(
            symmelem.transformations(),
            [
                Rotoreflection(vec, 1 / 3 * TAU),
                Rotation(vec, 2 / 3 * TAU),
                Reflection(vec),
                Rotation(vec, (4 % 3) / 3 * TAU),
                Rotoreflection(vec, (5 % 3) / 3 * TAU),
            ],
        )

    def test_symb(self):
        order = 3
        symmelem = RotoreflectionAxis(randvec(), order)
        self.assertEqual(symmelem.symb(), f"S{order}")

    def test_symmetric(self):
        symmelem = RotoreflectionAxis(randvec(), 3)
        points: List[Transformable] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in symmelem.transformations():
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Elems(points), TOL))


if __name__ == "__main__":
    main()
