from .const import *
from .vecop import *
from .transform import *
from .primitive import *
from .tools import *
from .ptgrp import *
from .chem import *
from .irrep import *

# add `inplane`, `inline` functions
# SymmOps to SymmElems, and save multiple cos and sin for all angles in Rotation
# add function `symmetric` to SymmElems
# add irreps, multiplication table of irreps, reduction of reps, dipole moments, subgroups, h, abelian
# rtol, atol
# def symb2group -> generate operations from few through transform
# ops2mats -> v,h,d; sort ops
# other primitives: segment, line, face, rotor
# Cooh, Doo, K
# i == S2
# class Group
# compare two structs with their distance mats
# decide wether to center elems in ptgrp because if a point not centered it will not be Kh
# change Transform in generate to SymmElem
# save a list of tried directions to avoid repetitions in symmelems
# determine the number of points on the other side for Sn using ABS(dist)
# symmetric electron maxima in Coo
