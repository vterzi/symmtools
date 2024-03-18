__all__ = ('vector', 'normalize', 'same', 'parallel', 'perpendicular',
           'translate', 'invert', 'translate2', 'rotate', 'reflect')

from numpy import array, sin, cos, dot, cross
from numpy.linalg import norm


def vector(vec):
    return array(vec, dtype=float)


def normalize(vec):
    return vec / norm(vec)


def same(vec1, vec2, tol):
    return norm(vec1 - vec2) <= tol


def parallel(vec1, vec2, tol):
    return norm(cross(vec1, vec2)) <= tol


def perpendicular(vec1, vec2, tol):
    return abs(dot(vec1, vec2)) <= tol


def translate(point, translation):
    return point + translation


def invert(point):
    return - point


def translate2(point, normal, coef1, coef2):
    base = dot(point, normal) * normal
    projection = point - base
    perpendicular_ = cross(normal, projection)
    return base + projection * coef1 + perpendicular_ * coef2


def rotate(point, rotation, angle):
    return translate2(point, rotation, cos(angle), sin(angle))


def reflect(point, reflection):
    return point - 2 * dot(point, reflection) * reflection
