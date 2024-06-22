from .init import (
    TestCase,
    main,
    randvec,
    randunitvec,
    List,
)

from symmtools import (
    TAU,
    TOL,
    SYMB,
    IdentityElement,
    InversionCenter,
    RotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    Inversion,
    Rotation,
    Reflection,
    Rotoreflection,
    Point,
    Points,
    normalize,
)


class TestIdentityElement(TestCase):
    def test_transformations(self) -> None:
        symmelem = IdentityElement()
        self.assertSequenceEqual(tuple(symmelem.transforms), [])

    def test_symb(self) -> None:
        symmelem = IdentityElement()
        self.assertEqual(symmelem.symb, "E")

    def test_symmetric(self) -> None:
        symmelem = IdentityElement()
        points = [Point(randvec()) for _ in range(3)]
        self.assertTrue(symmelem.symmetric(Points(points), TOL))


class TestInversionCenter(TestCase):
    def test_transformations(self) -> None:
        symmelem = InversionCenter()
        self.assertSequenceEqual(tuple(symmelem.transforms), [Inversion()])

    def test_symb(self) -> None:
        symmelem = InversionCenter()
        self.assertEqual(symmelem.symb, "i")

    def test_symmetric(self) -> None:
        symmelem = InversionCenter()
        points: List[Point] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in tuple(symmelem.transforms):
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Points(points), TOL))


class TestRotationAxis(TestCase):
    def test_init(self) -> None:
        self.assertRaises(ValueError, RotationAxis, randunitvec(), -1)
        self.assertRaises(ValueError, RotationAxis, randunitvec(), 0)
        self.assertRaises(ValueError, RotationAxis, randunitvec(), 1)

    def test_transformations(self) -> None:
        vec = randunitvec()
        symmelem = RotationAxis(vec, 3)
        vec = normalize(vec)
        self.assertSequenceEqual(
            tuple(symmelem.transforms),
            [Rotation(vec, 1 / 3 * TAU), Rotation(vec, 2 / 3 * TAU)],
        )
        vec = randunitvec()
        symmelem = RotationAxis(vec, 4)
        vec = normalize(vec)
        self.assertSequenceEqual(
            tuple(symmelem.transforms),
            [
                Rotation(vec, 1 / 4 * TAU),
                Rotation(vec, 2 / 4 * TAU),
                Rotation(vec, 3 / 4 * TAU),
            ],
        )

    def test_symb(self) -> None:
        order = 3
        symmelem = RotationAxis(randunitvec(), order)
        self.assertEqual(symmelem.symb, f"C{order}")

    def test_symmetric(self) -> None:
        symmelem = RotationAxis(randunitvec(), 3)
        points: List[Point] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in tuple(symmelem.transforms):
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Points(points), TOL))


class TestReflectionPlane(TestCase):
    def test_transformations(self) -> None:
        vec = randunitvec()
        symmelem = ReflectionPlane(vec)
        vec = normalize(vec)
        self.assertSequenceEqual(tuple(symmelem.transforms), [Reflection(vec)])

    def test_symb(self) -> None:
        symmelem = ReflectionPlane(randunitvec())
        self.assertEqual(symmelem.symb, SYMB.refl)

    def test_symmetric(self) -> None:
        symmelem = ReflectionPlane(randunitvec())
        points: List[Point] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in tuple(symmelem.transforms):
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Points(points), TOL))


class TestRotoreflectionAxis(TestCase):
    def test_init(self) -> None:
        self.assertRaises(ValueError, RotoreflectionAxis, randunitvec(), -1)
        self.assertRaises(ValueError, RotoreflectionAxis, randunitvec(), 0)
        self.assertRaises(ValueError, RotoreflectionAxis, randunitvec(), 1)
        self.assertRaises(ValueError, RotoreflectionAxis, randunitvec(), 2)

    def test_transformations(self) -> None:
        vec = randunitvec()
        symmelem = RotoreflectionAxis(vec, 3)
        vec = normalize(vec)
        self.assertSequenceEqual(
            tuple(symmelem.transforms),
            [
                Rotoreflection(vec, 1 / 3 * TAU),
                Rotation(vec, 2 / 3 * TAU),
                Reflection(vec),
                Rotation(vec, (4 % 3) / 3 * TAU),
                Rotoreflection(vec, (5 % 3) / 3 * TAU),
            ],
        )
        vec = randunitvec()
        symmelem = RotoreflectionAxis(vec, 4)
        vec = normalize(vec)
        self.assertSequenceEqual(
            tuple(symmelem.transforms),
            [
                Rotoreflection(vec, 1 / 4 * TAU),
                Rotation(vec, 2 / 4 * TAU),
                Rotoreflection(vec, 3 / 4 * TAU),
            ],
        )
        vec = randunitvec()
        symmelem = RotoreflectionAxis(vec, 5)
        vec = normalize(vec)
        self.assertSequenceEqual(
            tuple(symmelem.transforms),
            [
                Rotoreflection(vec, 1 / 5 * TAU),
                Rotation(vec, 2 / 5 * TAU),
                Rotoreflection(vec, 3 / 5 * TAU),
                Rotation(vec, 4 / 5 * TAU),
                Reflection(vec),
                Rotation(vec, (6 % 5) / 5 * TAU),
                Rotoreflection(vec, (7 % 5) / 5 * TAU),
                Rotation(vec, (8 % 5) / 5 * TAU),
                Rotoreflection(vec, (9 % 5) / 5 * TAU),
            ],
        )
        vec = randunitvec()
        symmelem = RotoreflectionAxis(vec, 6)
        vec = normalize(vec)
        self.assertSequenceEqual(
            tuple(symmelem.transforms),
            [
                Rotoreflection(vec, 1 / 6 * TAU),
                Rotation(vec, 2 / 6 * TAU),
                Inversion(),
                Rotation(vec, 4 / 6 * TAU),
                Rotoreflection(vec, 5 / 6 * TAU),
            ],
        )

    def test_symb(self) -> None:
        order = 3
        symmelem = RotoreflectionAxis(randunitvec(), order)
        self.assertEqual(symmelem.symb, f"S{order}")

    def test_symmetric(self) -> None:
        symmelem = RotoreflectionAxis(randunitvec(), 3)
        points: List[Point] = []
        for _ in range(3):
            points.append(Point(randvec()))
        for i in range(len(points)):
            for transform in tuple(symmelem.transforms):
                points.append(transform(points[i]))
        self.assertTrue(symmelem.symmetric(Points(points), TOL))


if __name__ == "__main__":
    main()
