"""
Implements Hyperbolic functions for arrays.
"""
# ===----------------------------------------------------------------------=== #
# Hyperbolic functions
# ===----------------------------------------------------------------------=== #

import math

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
import numojo.core.matrix as matrix

# TODO: add  dtype in backends and pass it here.

# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig
# ===------------------------------------------------------------------------===#


fn arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.acosh](A)


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
        The element-wise acosh of `array` in radians.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.acosh](
        array
    )


fn acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.acosh](A)


fn arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.asinh](A)


fn asinh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply asinh also known as inverse hyperbolic sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The element-wise asinh of `array` in radians.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.asinh](
        array
    )


fn asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.asinh](A)


fn arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.atanh](A)


fn atanh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply atanh also known as inverse hyperbolic tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The element-wise atanh of `array` in radians.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.atanh](
        array
    )


fn atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.atanh](A)


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
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The element-wise cosh of `array`.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.cosh](array)


fn cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.cosh](A)


fn sinh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply sin also known as hyperbolic sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The element-wise sinh of `array`.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.sinh](array)


fn sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.sinh](A)


fn tanh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply tan also known as hyperbolic tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The element-wise tanh of `array`.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.tanh](array)


fn tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return matrix._arithmetic_func_matrix_to_matrix[dtype, math.tanh](A)
