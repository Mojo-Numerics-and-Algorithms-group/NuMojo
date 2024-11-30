"""
Implements Matrix type (2-dimensional array)
"""

from .mat import Matrix
from .creation import full, zeros, ones, rand, fromstring, matrix
from .math import sum
from .linalg import matmul, solve, inv, det, trace, transpose, lstsq
from .stats import mean
from .sorting import sort
