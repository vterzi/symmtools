from re import fullmatch

from numpy import cross

from .const import NAN, PHI, TOL, PRIMAX, SECAX, INF_SYMB, REFL_SYMB
from .tools import signvar, ax3permut
from .symmelem import (
    SymmetryElement,
    IdentityElement,
    InversionCenter,
    RotationAxis,
    InfRotationAxis,
    ReflectionPlane,
    RotoreflectionAxis,
    InfRotoreflectionAxis,
)
from .primitive import Points
from .vecop import vector, norm, parallel, unitparallel, perpendicular
from .typehints import Union, Sequence, Tuple, List, Dict, Vector

_RotationAxis = Union[RotationAxis, InfRotationAxis]
_ReflectionPlane = ReflectionPlane
_RotoreflectionAxis = Union[RotoreflectionAxis, InfRotoreflectionAxis]


def symmelems(points: Points, tol: float = TOL) -> Tuple[
    int,
    bool,
    Sequence[_RotationAxis],
    Sequence[_ReflectionPlane],
    Sequence[_RotoreflectionAxis],
]:
    def contains(array: List[Vector], vector: Vector) -> bool:
        for elem in array:
            if unitparallel(elem, vector, tol):
                return True
        array.append(vector)
        return False

    def add_rotation(vector: Vector, order: int) -> bool:
        rotation = RotationAxis(vector, order)
        if rotation.symmetric(points, tol):
            i = 0
            while i < len(rotations) and rotations[i].order > order:
                i += 1
            rotations.insert(i, rotation)
            return True
        return False

    def add_reflection(vector: Vector) -> bool:
        reflection = ReflectionPlane(vector)
        if reflection.symmetric(points, tol):
            reflections.append(reflection)
            return True
        return False

    def add_rotoreflection(vector: Vector, order: int) -> bool:
        for factor in (2, 1):
            order_ = factor * order
            if order_ > 2:
                rotoreflection = RotoreflectionAxis(vector, order_)
                if rotoreflection.symmetric(points, tol):
                    i = 0
                    while (
                        i < len(rotoreflections)
                        and rotoreflections[i].order > order_
                    ):
                        i += 1
                    rotoreflections.insert(i, rotoreflection)
                    return True
        return False

    if not points.nondegen(tol):
        raise ValueError(
            "at least two identical elements in the instance of "
            + points.__class__.__name__
            + " for the given tolerance"
        )
    dim = 3
    invertible = InversionCenter().symmetric(points, tol)
    rotations: List[_RotationAxis] = []
    reflections: List[_ReflectionPlane] = []
    rotoreflections: List[_RotoreflectionAxis] = []
    n_points = len(points)
    if n_points == 1:
        dim = 0
    axes: List[Vector] = []
    planes: List[Vector] = []
    direction = None
    collinear = True
    for i1 in range(n_points - 1):
        # for all point pairs
        for i2 in range(i1 + 1, n_points):
            # for all point triplets
            for i3 in range(i2 + 1, n_points):
                # find the normal of the plane containing the point triplet
                normal: Vector = cross(
                    points[i2].pos - points[i1].pos,
                    points[i3].pos - points[i1].pos,
                )
                normal_norm = norm(normal)
                # if the point triplet is collinear
                if normal_norm <= tol:
                    continue
                # not all points are collinear
                collinear = False
                rotation = normal / normal_norm
                if not contains(axes, rotation):
                    # calculate the distance between the origin and the plane
                    dist = points[i1].pos.dot(rotation)
                    # initial number of points in the plane
                    max_order = 3
                    # for other points
                    for i4 in range(i3 + 1, n_points):
                        # if the point is in the plane
                        if abs(points[i4].pos.dot(rotation) - dist) <= tol:
                            # increase the number of points in the plane
                            max_order += 1
                    if (
                        max_order == n_points
                        and direction is None
                        and abs(dist) <= tol
                    ):
                        # all points are coplanar
                        dim = 2
                        direction = rotation
                        # add reflection (described by the plane containing
                        #   all points)
                        reflections.append(ReflectionPlane(rotation))
                    # for all possible orders > 2 starting from the highest
                    for order in range(max_order, 2, -1):
                        if add_rotoreflection(rotation, order):
                            break
                    # for all possible orders > 1 starting from the highest
                    for order in range(max_order, 1, -1):
                        if add_rotation(rotation, order):
                            break
            # directed segment between the point pair
            segment = points[i1].pos - points[i2].pos
            # midpoint of the segment
            midpoint = 0.5 * (points[i1].pos + points[i2].pos)
            reflection = segment / norm(segment)
            # add rotation with order infinity
            if (
                collinear
                and direction is None
                and parallel(points[i1].pos, points[i2].pos, tol)
            ):
                dim = 1
                direction = reflection
                rotations.insert(0, InfRotationAxis(reflection))
                if invertible:
                    rotoreflections.insert(
                        0, InfRotoreflectionAxis(reflection)
                    )
            # if the distance from the origin to the segment doesn't divide it
            #   in halves
            if not perpendicular(segment, midpoint, tol):
                continue
            midpoint_norm = norm(midpoint)
            # if the distance from the origin to the midpoint is not zero or
            #   all points are in one plane
            if midpoint_norm > tol or dim == 2:
                rotation = (
                    midpoint / midpoint_norm
                    if direction is None
                    else cross(direction, reflection)
                )
                if not contains(axes, rotation):
                    order = 2
                    add_rotoreflection(rotation, order)
                    add_rotation(rotation, order)
            if not contains(planes, reflection):
                add_reflection(reflection)
    return (
        dim,
        invertible,
        tuple(rotations),
        tuple(reflections),
        tuple(rotoreflections),
    )


def ptgrp(points: Points, tol: float = TOL) -> str:
    dim, invertible, rotations, reflections, rotoreflections = symmelems(
        points, tol
    )
    if dim == 0:
        return "Kh"  # 'K'
    if dim == 1:
        return f"D{INF_SYMB}h" if invertible else f"C{INF_SYMB}v"
    sigma = len(reflections) > 0
    if len(rotations) == 0:
        if sigma:
            return "Cs"
        if invertible:
            return "Ci"
        return "C1"
    rotation = rotations[0]
    order = rotations[0].order
    h = False
    if sigma:
        for reflection in reflections:
            if unitparallel(rotation.vec, reflection.vec, tol):
                h = True
                break
    if len(rotations) >= 2:
        if order > 2 and rotations[1].order > 2:
            if order == 5:
                return "Ih" if sigma else "I"
            if order == 4:
                return "Oh" if sigma else "O"
            return ("Th" if h else "Td") if sigma else "T"
        return (f"D{order}h" if h else f"D{order}d") if sigma else f"D{order}"
    else:
        if sigma:
            return f"C{order}h" if h else f"C{order}v"
        if len(rotoreflections) > 0:
            return f"S{2*order}"
        return f"C{order}"


def symb2grp(
    symb: str,
) -> Dict[str, Union[SymmetryElement, Sequence[SymmetryElement]]]:
    match = fullmatch(rf"([SCDTOIK])([1-9]\d*|{INF_SYMB})?([sivdh])?", symb)
    if match is None:
        raise ValueError("unknown point group")
    rotation, order, reflection = match.groups()
    inf = order == INF_SYMB
    n = int(order) if order and not inf else 0
    group: Dict[str, Union[SymmetryElement, Sequence[SymmetryElement]]] = {
        "E": IdentityElement()
    }
    if rotation == "S" and order and not reflection:
        if n % 2 == 1 or inf:
            return symb2grp(f"C{order}h")
        elif n == 2:
            return symb2grp("Ci")
        else:
            group[f"S{order}"] = RotoreflectionAxis(PRIMAX, n)
            group[f"C{n // 2}"] = RotationAxis(PRIMAX, n)
            if (n // 2) % 2 == 1:
                group["i"] = InversionCenter()
    elif rotation == "C":
        if order and not reflection:
            if n > 1:
                group[f"C{order}"] = RotationAxis(PRIMAX, n)
        elif not order and reflection == "s":
            group[REFL_SYMB] = ReflectionPlane(PRIMAX)
        elif reflection == "i":
            if not order or n == 1:
                group["i"] = InversionCenter()
            else:
                if n % 2 == 1:
                    return symb2grp(f"S{2 * n}")
                elif (n // 2) % 2 == 1:
                    return symb2grp(f"S{n // 2}")
                else:
                    return symb2grp(f"S{order}")
        elif order and reflection == "v":
            if n == 1:
                return symb2grp("Cs")
            else:
                vec = PRIMAX
                group[f"C{order}"] = RotationAxis(vec, n)
                s = ReflectionPlane(SECAX)
                if not inf:
                    transforms = RotationAxis(vec, 2 * n).transformations()
                    planes = (s,) + tuple(
                        transform(s) for transform in transforms[: n - 1]
                    )
                    if n % 2 == 1:
                        group[f"{REFL_SYMB}v"] = planes
                    else:
                        group[f"{REFL_SYMB}v"] = tuple(
                            planes[i] for i in range(0, len(planes), 2)
                        )
                        group[f"{REFL_SYMB}d"] = tuple(
                            planes[i] for i in range(1, len(planes), 2)
                        )
                else:
                    s._vec = vector([NAN, NAN, 0])
                    group[f"{REFL_SYMB}v"] = s
        elif order and reflection == "h":
            if n == 1:
                return symb2grp("Cs")
            else:
                group[f"C{order}"] = RotationAxis(PRIMAX, n)
                group[f"{REFL_SYMB}h"] = ReflectionPlane(PRIMAX)
                if n % 2 == 0:
                    group["i"] = InversionCenter()
                if n > 2:
                    group[f"S{order}"] = RotoreflectionAxis(PRIMAX, n)
        else:
            raise ValueError("unknown point group")
    elif rotation == "D" and order:
        vec = PRIMAX
        group[f"C{order}"] = RotationAxis(vec, n)
        C2 = RotationAxis(SECAX, 2)
        if not inf:
            transforms = RotationAxis(vec, 2 * n).transformations()
            axes = (C2,) + tuple(
                transform(C2) for transform in transforms[: n - 1]
            )
            if n % 2 == 1:
                group["C2'"] = axes
            else:
                group["C2'"] = tuple(axes[i] for i in range(0, len(axes), 2))
                group["C2''"] = tuple(axes[i] for i in range(1, len(axes), 2))
        else:
            C2._vec = vector([NAN, NAN, 0])
            group["C2'"] = C2
        if not reflection:
            if n == 1:
                return symb2grp(f"C{2 * n}")
            elif inf:
                group["i"] = InversionCenter()
        elif reflection == "d":
            if n == 1:
                return symb2grp(f"C{2 * n}h")
            elif inf:
                return symb2grp(f"D{order}h")
            else:
                s = ReflectionPlane(SECAX)
                transforms = RotationAxis(vec, 2 * n).transformations()
                planes = tuple(transforms[i](s) for i in range(1, n, 2))
                group[f"{REFL_SYMB}d"] = planes
                if n % 2 == 1:
                    group["i"] = InversionCenter()
                group[f"S{2 * n}"] = RotoreflectionAxis(PRIMAX, 2 * n)
        elif reflection == "h":
            if n == 1:
                return symb2grp(f"C{2 * n}v")
            else:
                group[f"{REFL_SYMB}h"] = ReflectionPlane(PRIMAX)
                s = ReflectionPlane(SECAX)
                if not inf:
                    transforms = RotationAxis(vec, 2 * n).transformations()
                    planes = (s,) + tuple(
                        transform(s) for transform in transforms[: n - 1]
                    )
                    if n % 2 == 1:
                        group[f"{REFL_SYMB}v"] = planes
                    else:
                        group[f"{REFL_SYMB}v"] = tuple(
                            planes[i] for i in range(0, len(planes), 2)
                        )
                        group[f"{REFL_SYMB}d"] = tuple(
                            planes[i] for i in range(1, len(planes), 2)
                        )
                        group["i"] = InversionCenter()
                    if n > 2:
                        group[f"S{order}"] = RotoreflectionAxis(PRIMAX, n)
                else:
                    s._vec = vector([NAN, NAN, 0])
                    group[f"{REFL_SYMB}v"] = s
        else:
            raise ValueError("unknown point group")
    elif rotation == "T" and not order:
        vecs3 = signvar([1, 1, 1], 1)
        vecs2 = ax3permut([[1]])
        for n, vecs in ((3, vecs3), (2, vecs2)):
            group[f"C{n}"] = tuple(RotationAxis(vec, n) for vec in vecs)
        if reflection == "d":
            group[f"{REFL_SYMB}d"] = tuple(
                ReflectionPlane(vec)
                for vec in ax3permut(signvar([1, 1], 0, True))
            )
            n = 4
            group[f"S{n}"] = tuple(RotoreflectionAxis(vec, n) for vec in vecs2)
        elif reflection == "h":
            group[f"{REFL_SYMB}h"] = tuple(
                ReflectionPlane(vec) for vec in vecs2
            )
            group["i"] = InversionCenter()
            n = 6
            group[f"S{n}"] = tuple(RotoreflectionAxis(vec, n) for vec in vecs3)
        elif reflection:
            raise ValueError("unknown point group")
    elif rotation == "O" and not order:
        vecs4 = ax3permut([[1]])
        vecs3 = signvar([1, 1, 1], 1)
        vecs2 = ax3permut(signvar([1, 1], 0, True))
        for n, vecs in ((4, vecs4), (3, vecs3), (2, vecs2)):
            group[f"C{n}"] = tuple(RotationAxis(vec, n) for vec in vecs)
        if reflection == "h":
            group[f"{REFL_SYMB}h"] = tuple(
                ReflectionPlane(vec) for vec in vecs4
            )
            group[f"{REFL_SYMB}d"] = tuple(
                ReflectionPlane(vec) for vec in vecs2
            )
            group["i"] = InversionCenter()
            for n, vecs in ((6, vecs3), (4, vecs4)):
                group[f"S{n}"] = tuple(
                    RotoreflectionAxis(vec, n) for vec in vecs
                )
        elif reflection:
            raise ValueError("unknown point group")
    elif rotation == "I" and not order:
        vecs5 = ax3permut(signvar([PHI, 1], 0, True))
        vecs3 = signvar([1, 1, 1], 1) + ax3permut(
            signvar([1, 1 + PHI], 0, True)
        )
        vecs2 = ax3permut([[1], *signvar([1, PHI, 1 + PHI], 0, True)])
        for n, vecs in ((5, vecs5), (3, vecs3), (2, vecs2)):
            group[f"C{n}"] = tuple(RotationAxis(vec, n) for vec in vecs)
        if reflection == "h":
            group[REFL_SYMB] = tuple(ReflectionPlane(vec) for vec in vecs2)
            group["i"] = InversionCenter()
            for n, vecs in ((10, vecs5), (6, vecs3)):
                group[f"S{n}"] = tuple(
                    RotoreflectionAxis(vec, n) for vec in vecs
                )
        elif reflection:
            raise ValueError("unknown point group")
    elif rotation == "K" and not order:
        Cn = InfRotationAxis(PRIMAX)
        Cn._vec = vector([NAN, NAN, NAN])
        group[f"C{INF_SYMB}"] = Cn
        if reflection == "h":
            s = ReflectionPlane(PRIMAX)
            s._vec = vector([NAN, NAN, NAN])
            group[REFL_SYMB] = s
            group["i"] = InversionCenter()
            Sn = InfRotoreflectionAxis(PRIMAX)
            Sn._vec = vector([NAN, NAN, NAN])
            group[f"S{INF_SYMB}"] = Sn
        elif reflection:
            raise ValueError("unknown point group")
    else:
        raise ValueError("unknown point group")
    return group
