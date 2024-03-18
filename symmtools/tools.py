__all__ = ('phi', 'eps', 'dtol', 'project', 'signpermut', 'ax3permut', 'topoints', 'generate', 'read')

from re import findall

from .primitive import Point, LabeledPoint, Elems

phi = (1 + 5 ** .5) / 2
eps = 7 / 3 - 4 / 3 - 1
dtol = 16 * eps

_label_re = r'(?:\b[A-Za-z_]\w*\b)'
_float_re = r'(?:[+\-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+\-]?\d+)?)'


def project(vecs, ref, axes=None):
    if axes is None:
        axes = tuple(range(len(ref)))
    res = []
    for vec in vecs:
        new = [*ref]
        for i, coord in enumerate(vec):
            new[axes[i]] += coord
        res.append(tuple(new))
    return tuple(res)


def signpermut(vec, parity=0, unique=False):
    vecs = []
    for permut in range(2 ** len(vec)):
        new = [*vec]
        sign = 1
        i = 0
        while permut > 0:
            if permut % 2 == 1:
                new[i] *= -1
                sign *= -1
            permut //= 2
            i += 1
        if sign * parity >= 0:
            if unique and tuple(- coord for coord in new) in vecs:
                continue
            vecs.append(tuple(new))
    return tuple(vecs)


def ax3permut(vecs):
    vecs = project(vecs, 3 * (0,))
    res = []
    for i in range(3):
        for vec in vecs:
            res.append((vec[i % 3], vec[(i + 1) % 3], vec[(i + 2) % 3]))
    return tuple(res)


def topoints(points):
    return tuple(Point(point) for point in points)


def generate(points, transforms=None, tol=dtol):
    points = list(points)
    fi = 0
    li = len(points)
    while fi < li:
        for transform in transforms:
            for i in range(fi, li):
                point = points[i].transform(transform)
                found = False
                for ref_point in points:
                    if point.same(ref_point, tol):
                        found = True
                        break
                if not found:
                    points.append(point)
        fi = li
        li = len(points)
    return Elems(points)


def read(string):
    elems = []
    for match in findall(fr'(?:({_label_re})\s+)?({_float_re})\s+({_float_re})\s+({_float_re})', string):
        label, x, y, z = match
        elems.append(LabeledPoint((x, y, z), label) if label else Point((x, y, z)))
    return Elems(elems)
