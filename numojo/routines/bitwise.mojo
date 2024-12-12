# ===------------------------------------------------------------------------===#
# Bit-wise operations
# ===------------------------------------------------------------------------===#

# ===------------------------------------------------------------------------===#
# Elementwise bit operations
# ===------------------------------------------------------------------------===#


import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant

import numojo.core._math_funcs as _mf
from ..core.ndarray import NDArray, NDArrayShape
from ..core.utility import is_inttype, is_floattype, is_booltype


fn invert[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise invert of an array.

    Constraints:
        The array must be either a boolean or integral array.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to the bitwise inversion of array.
    """
    constrained[
        is_inttype[dtype]() or is_booltype[dtype](),
        "Only Bools and integral types can be invertedd.",
    ]()

    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__invert__](
        array
    )
