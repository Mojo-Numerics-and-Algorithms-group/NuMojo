"""
Implements Hyperbolic functions for arrays.
"""
# ===----------------------------------------------------------------------=== #
# Hyperbolic functions
# ===----------------------------------------------------------------------=== #

import math

import numojo.core._math_funcs as _mf
from numojo.core.ndarray import NDArray

# TODO: add  dtype in backends and pass it here.

# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig
# ===------------------------------------------------------------------------===#


fn acosh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply acosh also known as inverse hyperbolic cosine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise acosh of `array` in radians.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.acosh](
        array
    )


fn asinh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply asinh also known as inverse hyperbolic sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise asinh of `array` in radians.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.asinh](
        array
    )


fn atanh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply atanh also known as inverse hyperbolic tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise atanh of `array` in radians.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.atanh](
        array
    )


# ===------------------------------------------------------------------------===#
# Hyperbolic Trig
# ===------------------------------------------------------------------------===#


fn cosh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply cosh also known as hyperbolic cosine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise cosh of `array`.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.cosh](array)


fn sinh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply sin also known as hyperbolic sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise sinh of `array`.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.sinh](array)


fn tanh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply tan also known as hyperbolic tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise tanh of `array`.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.tanh](array)
