# ===----------------------------------------------------------------------=== #
# NuMojo: Exponential routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Exponential routines for NuMojo (numojo.routines.math.exponents).

Implements element-wise exponential and logarithmic transformations for NDArrays.
"""

import std.math as math
from std.algorithm import parallelize
from std.algorithm import Static2DTileUnitFunc as Tile2DFunc
from std.utils import Variant

from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===------------------------------------------------------------------------===#
# Exponential functions
# ===------------------------------------------------------------------------===#


def exp[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise exponential of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is e**x for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.exp(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.exp](array)


def exp[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the exponential of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to e**value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var value: Scalar[f32] = 1.0
        var result = nm.exp(value)
        ```
    """
    return math.exp(value)


def exp2[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise base-2 exponential of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is 2**x for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.exp2(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.exp2](array)


def exp2[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the base-2 exponential of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to 2**value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var value: Scalar[f32] = 1.0
        var result = nm.exp2(value)
        ```
    """
    return math.exp2(value)


def expm1[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise exp(x) - 1 of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is exp(x) - 1 for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.expm1(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.expm1](array)


def expm1[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute exp(value) - 1 for a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to exp(value) - 1.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var value: Scalar[f32] = 1.0
        var result = nm.expm1(value)
        ```
    """
    return math.expm1(value)


# ===------------------------------------------------------------------------===#
# Logarithmic functions
# ===------------------------------------------------------------------------===#


def log[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise natural logarithm of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is ln(x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var arr = nm.arange[f64](1.0, 10.0, 1.0)
        var result = nm.log(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.log](array)


def log[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the natural logarithm of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to ln(value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var result = nm.log(10.0)
        ```
    """
    return math.log(value)


def log2[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise base-2 logarithm of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is log2(x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.arange[f64](1.0, 10.0, 1.0)
        var result = nm.log2(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.log2](array)


def log2[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the base-2 logarithm of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to log2(value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var result = nm.log2(10.0)
        ```
    """
    return math.log2(value)


def log10[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise base-10 logarithm of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is log10(x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var arr = nm.arange[f64](1.0, 10.0, 1.0)
        var result = nm.log10(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.log10](array)


def log10[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the base-10 logarithm of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to log10(value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var result = nm.log10(10.0)
        ```
    """
    return math.log10(value)


def log1p[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise ln(1 + x) of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is ln(1 + x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.log1p(arr)
        ```
    """
    return HostExecutor.apply_unary[dtype, math.log1p](array)


def log1p[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute ln(1 + value) for a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to ln(1 + value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var result = nm.log1p(1.0)
        ```
    """
    return math.log1p(value)
