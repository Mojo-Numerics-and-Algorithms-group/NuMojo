"""
`numojo.mat` sub-package provides:

- `Matrix` type (2DArray) with basic dunder methods and manipulation methods.
- Auxiliary types, e.g., `_MatrixIter` for iteration.
- Indexing and slicing.
- Arithmetic functions for item-wise calculation and broadcasting.
- Creation routines and functions to construct `Matrix` from other data objects, 
e.g., `List`, `NDArray`, `String`, and `numpy` array (in `creation` module).
- Linear algebra, e.g., matrix multiplication, decomposition, inverse of matrix, 
solve of linear system, Ordinary Least Square, etc (in `linalg` module).
- Mathematical functions, e.g., `sum` (in `math` module).
- Statistical functions, e.g., `mean` (in `stats` module).
- Sorting and searching, .e.g, `sort`, `argsort` (in `sorting` module).

TODO: In future, we can also make use of the trait `ArrayLike` to align
the behavior of `NDArray` type and the `Matrix` type.
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
    round,
    sum,
    prod,
    cumsum,
    cumprod,
)
from .logic import all, any
from .linalg import matmul, solve, inv, det, trace, transpose, lstsq
from .statistics import mean, variance, std
from .sorting import sort, argsort, max, argmax, min, argmin
