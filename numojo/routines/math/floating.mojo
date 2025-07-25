# ===------------------------------------------------------------------------===#
# Floating point routines
# ===------------------------------------------------------------------------===#

import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray


fn copysign[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Copy the sign of the first NDArray and apply it to the second NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The second NDArray multipied by the sign of the first NDArray.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, math.copysign](
        array1, array2
    )
