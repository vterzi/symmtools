from .init import (
    TestCase,
    main,
    normalvariate,
    clip,
    zeros,
    linear_sum_assignment,
    randint,
    randvec,
    TOL,
)

from symmtools.utils import (
    clamp,
    rational,
    signvar,
    circshift,
    linassign,
)


class TestUtils(TestCase):
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

    def test_signvar(self) -> None:
        arr = zeros(3)
        for i in range(len(arr)):
            while arr[i] == 0.0:
                arr[i] = normalvariate(0.0, 1.0)
        x, y, z = arr
        vecs = set(tuple(elem) for elem in signvar(tuple(arr)))
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
        vecs = set(tuple(elem) for elem in signvar(tuple(arr), 0, True))
        self.assertSetEqual(
            vecs, {(x, y, z), (-x, y, z), (x, -y, z), (-x, -y, z)}
        )
        vecs = set(tuple(elem) for elem in signvar(tuple(arr), 1))
        self.assertSetEqual(
            vecs, {(x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z)}
        )
        vecs = set(tuple(elem) for elem in signvar(tuple(arr), -1))
        self.assertSetEqual(
            vecs, {(-x, y, z), (x, -y, z), (x, y, -z), (-x, -y, -z)}
        )
        arr = zeros(2)
        vecs = set(tuple(elem) for elem in signvar(tuple(arr)))
        self.assertSetEqual(vecs, {(0.0, 0.0)})
        for i in range(len(arr)):
            while arr[i] == 0.0:
                arr[i] = normalvariate(0.0, 1.0)
        x, y = arr
        arr[0] = 0.0
        vecs = set(tuple(elem) for elem in signvar(tuple(arr)))
        self.assertSetEqual(vecs, {(0.0, y), (0.0, -y)})
        arr[0] = x
        arr[1] = 0.0
        vecs = set(tuple(elem) for elem in signvar(tuple(arr)))
        self.assertSetEqual(vecs, {(x, 0.0), (-x, 0.0)})
        arr[1] = y
        vecs = set(tuple(elem) for elem in signvar(tuple(arr)))
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

    def test_linassign(self) -> None:
        n = 8
        mat = tuple(tuple(randint(-n, n) for _ in range(n)) for _ in range(n))
        assignment1 = linassign(mat, fast=False)
        cost1 = sum(mat[i][assignment1[i]] for i in range(n))
        assignment2 = linear_sum_assignment(mat)[1]
        cost2 = sum(mat[i][assignment2[i]] for i in range(n))
        self.assertEqual(cost1, cost2)
        assignment1 = linassign(mat, True, fast=False)
        cost1 = sum(mat[i][assignment1[i]] for i in range(n))
        assignment2 = linear_sum_assignment(mat, True)[1]
        cost2 = sum(mat[i][assignment2[i]] for i in range(n))
        self.assertEqual(cost1, cost2)


if __name__ == "__main__":
    main()
