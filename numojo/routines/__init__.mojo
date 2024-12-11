"""
Implements routines by topic:

- Array creation routines (creation.mojo)
- Array manipulation routines (manipulation.mojo)
- Bit-wise operations (bitwise.mojo)
- Constants (constants.mojo)
- Linear algebra (linalg/)
    - Decompositions (decompositions.mojo)
    - Products of matrices and vectors (products.mojo)
    - Solving (solving.mojo)
- Logic functions (logic/)
- Mathematical functions (math/)
    - Backend functions for mathematics (_math_funcs.mojo)
    - ...
- Random sampling (random.mojo)
- Sorting, searching, and counting (sorting.mojo, searching.mojo)
- Statistics (statistics/)
    - Averages and variances (averages.mojo)

"""

from .io import *
from .linalg import *
from .logic import *
from .math import *
from .statistics import *
from .bitwise import *
from .constants import Constants
from .creation import *
from .manipulation import *
from .random import *
from .sorting import *
from .searching import argmax, argmin
