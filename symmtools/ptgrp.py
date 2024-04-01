from re import fullmatch

from numpy import nan, inf, array, dot, cross
from numpy.linalg import norm

from .const import PHI, TOL
from .tools import signpermut, ax3permut
from .transform import Identity, Inversion, Rotation, Reflection, Rotoreflection
from .vecop import parallel, perpendicular

primax = (0, 0, 1)
secax = (1, 0, 0)


def symb2grp(symb):
    match = fullmatch(r'([SCDTOIK])([1-9]\d*|oo)?([sivdh])?', symb)
    if match is None:
        raise ValueError('unknown point group')
    rotation, order, reflection = match.groups()
    n = (int(order) if order != 'oo' else inf) if order else nan
    group = {'E': Identity()}
    if rotation == 'S' and order and not reflection:
        if n % 2 == 1 or n == inf:
            return symb2grp(f'C{order}h')
        elif n == 2:
            return symb2grp('Ci')
        else:
            group[f'S{order}'] = Rotoreflection(primax, n)
            group[f'C{n // 2}'] = Rotation(primax, n)
            if (n // 2) % 2 == 1:
                group['i'] = Inversion()
    elif rotation == 'C':
        if order and not reflection:
            if n > 1:
                group[f'C{order}'] = Rotation(primax, n)
        elif not order and reflection == 's':
            group['s'] = Reflection(primax)
        elif reflection == 'i':
            if not order or n == 1:
                group['i'] = Inversion()
            else:
                if n % 2 == 1:
                    return symb2grp(f'S{2 * n}')
                elif (n // 2) % 2 == 1:
                    return symb2grp(f'S{n // 2}')
                else:
                    return symb2grp(f'S{order}')
        elif order and reflection == 'v':
            if n == 1:
                return symb2grp('Cs')
            else:
                vec = primax
                group[f'C{order}'] = Rotation(vec, n)
                s = Reflection(secax)
                if n < inf:
                    dn = 2 * n
                    if n % 2 == 1:
                        group['sv'] = (s,) + tuple(s.rotate(Rotation(vec, dn / factor))
                                                   for factor in range(1, n))
                    else:
                        group['sv'] = (s,) + tuple(s.rotate(Rotation(vec, dn / factor))
                                                   for factor in range(2, n, 2))
                        group['sd'] = tuple(s.rotate(Rotation(vec, dn / factor))
                                            for factor in range(1, n, 2))
                else:
                    s._vec = array([nan, nan, 0])
                    group['sv'] = s
        elif order and reflection == 'h':
            if n == 1:
                return symb2grp('Cs')
            else:
                group[f'C{order}'] = Rotation(primax, n)
                group['sh'] = Reflection(primax)
                if n % 2 == 0:
                    group['i'] = Inversion()
                if n > 2:
                    group[f'S{order}'] = Rotoreflection(primax, n)
        else:
            raise ValueError('unknown point group')
    elif rotation == 'D' and order:
        vec = primax
        group[f'C{order}'] = Rotation(vec, n)
        C2 = Rotation(secax, 2)
        if n < inf:
            dn = 2 * n
            if n % 2 == 1:
                group["C2'"] = (C2,) + tuple(C2.rotate(Rotation(vec, dn / factor))
                                             for factor in range(1, n))
            else:
                group["C2'"] = (C2,) + tuple(C2.rotate(Rotation(vec, dn / factor))
                                             for factor in range(2, n, 2))
                group["C2''"] = tuple(C2.rotate(Rotation(vec, dn / factor))
                                      for factor in range(1, n, 2))
        else:
            C2._vec = array([nan, nan, 0])
            group["C2'"] = C2
        if not reflection:
            if n == 1:
                return symb2grp(f'C{2 * n}')
            elif n == inf:
                group['i'] = Inversion()
        elif reflection == 'd':
            if n == 1:
                return symb2grp(f'C{2 * n}h')
            elif n == inf:
                return symb2grp(f'D{order}h')
            else:
                s = Reflection(secax)
                group['sd'] = tuple(s.rotate(Rotation(vec, 2 * n / factor))
                                    for factor in range(1, n, 2))
                if n % 2 == 1:
                    group['i'] = Inversion()
                group[f'S{2 * n}'] = Rotoreflection(primax, 2 * n)
        elif reflection == 'h':
            if n == 1:
                return symb2grp(f'C{2 * n}v')
            else:
                group['sh'] = Reflection(primax)
                s = Reflection(secax)
                if n < inf:
                    if n % 2 == 1:
                        group['sv'] = (s,) + tuple(s.rotate(Rotation(vec, 2 * n / factor))
                                                   for factor in range(1, n))
                    else:
                        group['sv'] = (s,) + tuple(s.rotate(Rotation(vec, 2 * n / factor))
                                                   for factor in range(2, n, 2))
                        group['sd'] = tuple(s.rotate(Rotation(vec, 2 * n / factor))
                                            for factor in range(1, n, 2))
                        group['i'] = Inversion()
                    if n > 2:
                        group[f'S{order}'] = Rotoreflection(primax, n)
                else:
                    s._vec = array([nan, nan, 0])
                    group['sv'] = s
        else:
            raise ValueError('unknown point group')
    elif rotation == 'T' and not order:
        vecs3 = signpermut([1, 1, 1], 1)
        vecs2 = ax3permut([[1]])
        for n, vecs in ((3, vecs3), (2, vecs2)):
            group[f'C{n}'] = tuple(Rotation(vec, n) for vec in vecs)
        if reflection == 'd':
            group['sd'] = tuple(Reflection(vec) for vec in ax3permut(signpermut([1, 1], 0, True)))
            n = 4
            group[f'S{n}'] = tuple(Rotoreflection(vec, n) for vec in vecs2)
        elif reflection == 'h':
            group['sh'] = tuple(Reflection(vec) for vec in vecs2)
            group['i'] = Inversion()
            n = 6
            group[f'S{n}'] = tuple(Rotoreflection(vec, n) for vec in vecs3)
        elif reflection:
            raise ValueError('unknown point group')
    elif rotation == 'O' and not order:
        vecs4 = ax3permut([[1]])
        vecs3 = signpermut([1, 1, 1], 1)
        vecs2 = ax3permut(signpermut([1, 1], 0, True))
        for n, vecs in ((4, vecs4), (3, vecs3), (2, vecs2)):
            group[f'C{n}'] = tuple(Rotation(vec, n) for vec in vecs)
        if reflection == 'h':
            group['sh'] = tuple(Reflection(vec) for vec in vecs4)
            group['sd'] = tuple(Reflection(vec) for vec in vecs2)
            group['i'] = Inversion()
            for n, vecs in ((6, vecs3), (4, vecs4)):
                group[f'S{n}'] = tuple(Rotoreflection(vec, n) for vec in vecs)
        elif reflection:
            raise ValueError('unknown point group')
    elif rotation == 'I' and not order:
        vecs5 = ax3permut(signpermut([PHI, 1], 0, True))
        vecs3 = signpermut([1, 1, 1], 1) + ax3permut(signpermut([1, 1 + PHI], 0, True))
        vecs2 = ax3permut([[1], *signpermut([1, PHI, 1 + PHI], 0, True)])
        for n, vecs in ((5, vecs5), (3, vecs3), (2, vecs2)):
            group[f'C{n}'] = tuple(Rotation(vec, n) for vec in vecs)
        if reflection == 'h':
            group['s'] = tuple(Reflection(vec) for vec in vecs2)
            group['i'] = Inversion()
            for n, vecs in ((10, vecs5), (6, vecs3)):
                group[f'S{n}'] = tuple(Rotoreflection(vec, n) for vec in vecs)
        elif reflection:
            raise ValueError('unknown point group')
    elif rotation == 'K' and not order:
        Cn = Rotation(primax, inf)
        Cn._vec = array([nan, nan, nan])
        group['Coo'] = Cn
        if reflection == 'h':
            s = Reflection(primax)
            s._vec = array([nan, nan, nan])
            group['s'] = s
            group['i'] = Inversion()
            Sn = Rotoreflection(primax, inf)
            Sn._vec = array([nan, nan, nan])
            group['Soo'] = Sn
        elif reflection:
            raise ValueError('unknown point group')
    else:
        raise ValueError('unknown point group')
    return group


def symmelems(elems, tol=TOL):
    def contains(array, vector):
        for elem in array:
            if parallel(elem.vec, vector, tol):
                return True
        return False

    def add_rotation(vector, order_):
        # for all angles
        for factor in range(1, order_):
            # if not symmetric for an angle
            if not elems.same(elems.rotate(Rotation(vector, order_ / factor)), tol):
                return False
        rotations.append(Rotation(vector, order_))
        return True

    def add_reflection(vector):
        reflection_ = Reflection(vector)
        if not elems.same(elems.reflect(reflection_), tol):
            return False
        reflections.append(reflection_)
        return True

    def add_rotoreflection(vector, order_):
        # check the doubled order first
        for step in (2, 1):
            order__ = order_ * step
            if order__ == 2:
                continue
            symmetric = True
            # for all angles (only odd factors in the case of doubled order)
            for factor in range(1, order__, step):
                if not elems.same(elems.rotoreflect(Rotoreflection(vector, order__ / factor)), tol):
                    symmetric = False
                    break
            # if symmetric for all angles
            if symmetric:
                rotoreflections.append(Rotoreflection(vector, order__))
                return True
        return False

    if not elems.check(tol):
        raise ValueError('at least two identical elements in the instance of Struct for given tolerance')
    dim = 3
    invertible = elems.same(elems.invert(), tol)
    rotations = []
    reflections = []
    rotoreflections = []
    if len(elems) == 1:
        dim = 0
    direction = None
    collinear = True
    for i1 in range(len(elems) - 1):
        # for all point pairs
        for i2 in range(i1 + 1, len(elems)):
            # for all point triplets
            for i3 in range(i2 + 1, len(elems)):
                # find the normal of the plane containing the point triplet
                normal = cross(elems[i2].pos - elems[i1].pos, elems[i3].pos - elems[i1].pos)
                len_normal = norm(normal)
                # if the point triplet is collinear
                if len_normal <= tol:
                    continue
                # not all points are collinear
                collinear = False
                rotation = normal / len_normal
                if not contains(rotations, rotation):
                    # calculate the distance between the origin and the plane
                    dist = dot(elems[i1].pos, rotation)
                    # initial number of points in the plane
                    max_order = 3
                    # for other points
                    for i4 in range(i3 + 1, len(elems)):
                        # if the point is in the plane
                        if abs(dot(elems[i4].pos, rotation) - dist) <= tol:
                            # increase the number of points in the plane
                            max_order += 1
                    if max_order == len(elems) and direction is None and abs(dist) <= tol:
                        # all points are coplanar
                        dim = 2
                        direction = rotation
                        # add reflection (described by the plane containing all points)
                        reflections.append(Reflection(rotation))
                    added = False
                    # for all possible orders > 2 starting from the highest
                    for order in range(max_order, 2, -1):
                        added |= add_rotation(rotation, order)
                        added |= add_rotoreflection(rotation, order)
                        if added:
                            break
            # directed segment between the point pair
            segment = elems[i1].pos - elems[i2].pos
            # midpoint of the segment
            midpoint = (elems[i1].pos + elems[i2].pos) / 2
            reflection = segment / norm(segment)
            # add rotation with order infinity
            if collinear and direction is None and parallel(elems[i1].pos, elems[i2].pos, tol):
                dim = 1
                direction = reflection
                rotations.append(Rotation(reflection, inf))
                if invertible:
                    rotoreflections.append(Rotoreflection(reflection, inf))
            # if the distance from the origin to the segment doesn't divide it in halves
            if not perpendicular(segment, midpoint, tol):
                continue
            len_midpoint = norm(midpoint)
            # if the distance from the origin to the midpoint is not zero or all points are in one plane
            if len_midpoint > tol or dim == 2:
                rotation = midpoint / len_midpoint if direction is None else cross(direction, reflection)
                if not contains(rotations, rotation):
                    order = 2
                    add_rotation(rotation, order)
                    add_rotoreflection(rotation, order)
            if not contains(reflections, reflection):
                add_reflection(reflection)
    return (dim,
            invertible,
            tuple(sorted(rotations, key=lambda elem: - elem.order)),
            tuple(reflections),
            tuple(sorted(rotoreflections, key=lambda elem: - elem.order)))


def ptgrp(elems, tol=TOL):
    dim, invertible, rotations, reflections, rotoreflections = symmelems(elems, tol)
    if dim == 0:
        return 'Kh'  # 'K'
    if dim == 1:
        return 'Dooh' if invertible else 'Coov'
    if len(rotations) == 0:
        if len(reflections) > 0:
            return 'Cs'
        if invertible:
            return 'Ci'
        return 'C1'
    rotation = rotations[0]
    order = rotations[0].order
    sigma = False
    h = False
    if len(reflections) > 0:
        sigma = True
        for reflection in reflections:
            if parallel(rotation.vec, reflection.vec, tol):
                h = True
                break
    if len(rotations) >= 2:
        if order > 2 and rotations[1].order > 2:
            if order == 5:
                return 'Ih' if sigma else 'I'
            if order == 4:
                return 'Oh' if sigma else 'O'
            return ('Th' if h else 'Td') if sigma else 'T'
        return (f'D{order}h' if h else f'D{order}d') if sigma else f'D{order}'
    else:
        if sigma:
            return f'C{order}h' if h else f'C{order}v'
        if len(rotoreflections) > 0:
            return f'S{2*order}'
        return f'C{order}'
