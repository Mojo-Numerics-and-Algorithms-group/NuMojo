# ===----------------------------------------------------------------------=== #
# NuMojo: Datatype utilities
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Data type utility functions (numojo.core.dtype.utility)

This module provides utility functions for checking properties of data types (DType) at both compile time and run time.
"""


@parameter
def is_inttype[dtype: DType]() -> Bool:
    """
    Check if the given dtype is an integer type at compile time.

    Parameters:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is an integer type, False otherwise.
    """

    comptime if (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        return True
    return False


def is_inttype(dtype: DType) -> Bool:
    """
    Check if the given dtype is an integer type at run time.

    Args:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is an integer type, False otherwise.
    """
    if (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        return True
    return False


@parameter
def is_floattype[dtype: DType]() -> Bool:
    """
    Check if the given dtype is a floating point type at compile time.

    Parameters:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a floating point type, False otherwise.
    """

    comptime if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        return True
    return False


def is_floattype(dtype: DType) -> Bool:
    """
    Check if the given dtype is a floating point type at run time.

    Args:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a floating point type, False otherwise.
    """
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        return True
    return False


@parameter
def is_booltype[dtype: DType]() -> Bool:
    """
    Check if the given dtype is a boolean type at compile time.

    Parameters:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a boolean type, False otherwise.
    """

    comptime if dtype == DType.bool:
        return True
    return False


def is_booltype(dtype: DType) -> Bool:
    """
    Check if the given dtype is a boolean type at run time.

    Args:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a boolean type, False otherwise.
    """
    if dtype == DType.bool:
        return True
    return False
