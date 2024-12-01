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
from .mathematics import sum
from .linalg import matmul, solve, inv, det, trace, transpose, lstsq
from .statistics import mean
from .sorting import sort, argsort
