# ===----------------------------------------------------------------------=== #
# NuMojo: Hyperbolic routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Hyperbolic routines for NuMojo (numojo.routines.math.hyper).

Implements hyperbolic and inverse hyperbolic functions for NDArrays and Matrices.
"""

import std.math as math

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.core.matrix.base import _arithmetic_func_matrix_to_matrix

# TODO: add dtype in backends and pass it here.

# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig (NDArray)
# ===------------------------------------------------------------------------===#


fn acosh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise acosh of `array`.
    """
    return HostExecutor.apply_unary[dtype, math.acosh](array)


fn asinh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse hyperbolic sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise asinh of `array`.
    """
    return HostExecutor.apply_unary[dtype, math.asinh](array)


fn atanh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse hyperbolic tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise atanh of `array`.
    """
    return HostExecutor.apply_unary[dtype, math.atanh](array)


# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig (Matrix)
# ===------------------------------------------------------------------------===#


fn arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic cosine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic cosine (arccosh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.acosh](A)


fn acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic cosine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic cosine (acosh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.acosh](A)


fn arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic sine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic sine (arcsinh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.asinh](A)


fn asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic sine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic sine (asinh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.asinh](A)


fn arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic tangent element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic tangent (arctanh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.atanh](A)


fn atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic tangent element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic tangent (atanh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.atanh](A)


# ===------------------------------------------------------------------------===#
# Hyperbolic Trig (NDArray)
# ===------------------------------------------------------------------------===#


fn cosh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise cosh of `array`.
    """
    return HostExecutor.apply_unary[dtype, math.cosh](array)


fn sinh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hyperbolic sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise sinh of `array`.
    """
    return HostExecutor.apply_unary[dtype, math.sinh](array)


fn tanh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hyperbolic tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise tanh of `array`.
    """
    return HostExecutor.apply_unary[dtype, math.tanh](array)


# ===------------------------------------------------------------------------===#
# Hyperbolic Trig (Matrix)
# ===------------------------------------------------------------------------===#


fn cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise cosh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.cosh](A)


fn sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic sin.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise sinh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.sinh](A)


fn tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic tan.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise tanh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.tanh](A)
