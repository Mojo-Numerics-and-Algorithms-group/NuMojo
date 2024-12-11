# ===------------------------------------------------------------------------===#
# Logarithms
# ===------------------------------------------------------------------------===#


import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant

import numojo.core._math_funcs as _mf
from numojo.core.ndarray import NDArray


fn log[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise natural logarithm of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ln(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.log](array)


alias ln = log
"""
Natural Log equivelent to log
"""


fn log2[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise logarithm base two of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to log_2(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.log2](array)


fn log10[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise logarithm base ten of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to log_10(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.log10](
        array
    )


fn log1p[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise natural logarithm of 1 plus NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ln(NDArray+1).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.log1p](
        array
    )
