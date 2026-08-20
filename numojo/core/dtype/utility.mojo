# ===----------------------------------------------------------------------=== #
# NuMojo: Datatype utilities
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Datatype Utilities (numojo.core.dtype.utility).
===============================================

Type checking utilities for DType inspection.

Functions for checking properties of data types (DType) at both compile time
and runtime.

Exports
-------
- `is_inttype`: Check if DType is integer.
- `is_floattype`: Check if DType is floating-point.
- `is_complextype`: Check if DType is complex.
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
