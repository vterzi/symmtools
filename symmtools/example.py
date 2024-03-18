from .primitive import Point, Elems
from .tools import phi, dtol, project, signpermut, ax3permut, topoints, generate
from .transform import Rotation

point = Elems(topoints(project([[]], 3 * [0])))
segment = Elems(topoints(project(signpermut([1]), 3 * [0])))
triangle = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 3)], dtol)
square = Elems(topoints(project(signpermut([1, 1]), 3 * [0])))
pentagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 5)], dtol)
hexagon = generate([Point([1, 0, 0])], [Rotation([0, 0, 1], 6)], dtol)
tetrahedron = Elems(topoints(signpermut([1, 1, 1], 1)))
cube = Elems(topoints(signpermut([1, 1, 1])))
octahedron = Elems(topoints(ax3permut(signpermut([1]))))
icosahedron = Elems(topoints(ax3permut(signpermut([phi, 1]))))
dodecahedron = Elems(topoints(signpermut([phi, phi, phi]) + ax3permut(signpermut([phi + 1, 1]))))
