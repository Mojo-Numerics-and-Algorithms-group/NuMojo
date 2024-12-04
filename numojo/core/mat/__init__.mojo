"""
Implements Matrix type (2-dimensional array)
"""

from .matrix import Matrix
from .creation import (
    full,
    zeros,
    ones,
    rand,
    fromstring,
    fromlist,
)
from .mathematics import (
    sin,
    cos,
    tan,
    arcsin,
    asin,
    arccos,
    acos,
    arctan,
    atan,
    sinh,
    cosh,
    tanh,
    arcsinh,
    asinh,
    arccosh,
    acosh,
    arctanh,
    atanh,
    sum,
    prod,
)
from .logic import all, any
from .linalg import matmul, solve, inv, det, trace, transpose, lstsq
from .statistics import mean, variance, std
from .sorting import sort, argsort, max
