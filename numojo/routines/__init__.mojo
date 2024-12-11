"""
Implements routines by topic:

- Array creation routines (creation.mojo)
- Array manipulation routines (manipulation.mojo)
- Bit-wise operations (bitwise.mojo)
- Constants (constants.mojo)
- Input and output (io/)
    - Text files (files.mojo)
    - Text formatting options (formatting.mojo)
- Linear algebra (linalg/)
    - Decompositions (decompositions.mojo)
    - Products of matrices and vectors (products.mojo)
    - Solving (solving.mojo)
- Logic functions (logic/)
    - Comparison (comparison.mojo)
    - Array contents (contents.mojo)
    - Truth value testing (truth.mojo)
- Mathematical functions (math/)
    - Arithmetic operations (arithmetic.mojo)
    - Exponents and logarithms (exp_log.mojo)
    - Extrema finding (extrema.mojo)
    - Floating point routines (floating.mojo)
    - Hyperbolic functions (hyper.mojo)
    - Miscellaneous (misc.mojo)
    - Rounding (rounding.mojo)
    - Sums, products, differences (sums.mojo, products.mojo, differences.mojo)
    - Trigonometric functions (trig.mojo)
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
