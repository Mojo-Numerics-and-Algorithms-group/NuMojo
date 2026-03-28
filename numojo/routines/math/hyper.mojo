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


def acosh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
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


def asinh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
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


def atanh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
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


def arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
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


def acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
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


def arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
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


def asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
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


def arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
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


def atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
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


def cosh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
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


def sinh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
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


def tanh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
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


def cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise cosh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.cosh](A)


def sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic sin.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise sinh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.sinh](A)


def tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic tan.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise tanh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, math.tanh](A)
