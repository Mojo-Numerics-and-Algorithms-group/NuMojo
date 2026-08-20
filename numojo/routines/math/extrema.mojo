# ===----------------------------------------------------------------------=== #
# NuMojo: Extrema routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""
Extrema routines (numojo.routines.math.extrema).
================================================
Minimum and maximum operations for arrays.

Element-wise min/max comparisons and axis-aware reduction operations
for NDArrays and Matrices.

Exports
-------
- `min`, `max`: Element-wise minimum and maximum.
- `minimum`, `maximum`: Element-wise operations (aliases).
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math.math as stdlib_math
from max.algorithm import parallelize
from std.algorithm import vectorize
from std.collections.optional import Optional
from std.math import max as builtin_max
from std.math import min as builtin_min
from std.sys import simd_width_of

from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor
from numojo.routines.creation import full
from numojo.routines.sorting import binary_sort
from numojo.routines.functional import apply_along_axis_reduce
from numojo.routines.manipulation import ravel


# ===-----------------------------------------------------------------------===#
# NDArray reductions (min/max over axes)
# ===-----------------------------------------------------------------------===#


def extrema_1d[
    dtype: DType, //, is_max: Bool
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """
    Find the max or min value in the buffer.

    The input is treated as a 1-D array regardless of shape. This is the
    backend routine for `max` and `min`.

    Parameters:
        dtype: The element type.
        is_max: If True, find max value, otherwise find min value.

    Args:
        a: An array.

    Returns:
        The extreme value.
    """

    if not a.is_c_contiguous():
        return extrema_1d[is_max](a.contiguous())

    comptime simd_width = builtin_max(simd_width_of[dtype](), 64)
    var value = a.unsafe_load[width=1](0)

    comptime if is_max:

        def vectorize_max[simd_width: Int](offset: Int) {mut value, a}:
            var temp = a.unsafe_load[width=simd_width](offset).reduce_max()
            if temp >= value:
                value = temp

        vectorize[simd_width](a.size, vectorize_max)

        return value

    else:

        def vectorize_min[simd_width: Int](offset: Int) {mut value, a} -> None:
            var temp = a.unsafe_load[width=simd_width](offset).reduce_min()
            if temp < value:
                value = temp

        vectorize[simd_width](a.size, vectorize_min)

        return value


def max[dtype: DType](a: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Find the max value of an array.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The max value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.arange[f32](0, 6).reshape(Shape(2, 3))
        var m = nm.max(a)
        ```
    """

    if a.ndim == 1:
        return extrema_1d[is_max=True](a)
    else:
        return extrema_1d[is_max=True](ravel(a))


def extrema_1d_max[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """
    Find the max value in a 1-D array.
    """
    return extrema_1d[is_max=True](a)


def max[dtype: DType](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Find the max value of an array along an axis.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the max is performed.

    Returns:
        An array with reduced number of dimensions.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.arange[f32](0, 6).reshape(Shape(2, 3))
        var m = nm.max(a, axis=0)
        ```
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            String("Error in `max`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    return apply_along_axis_reduce[dtype, func1d=extrema_1d_max](
        a=a, axis=normalized_axis
    )


def min[dtype: DType](a: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Find the min value of an array.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The min value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.arange[f32](0, 6).reshape(Shape(2, 3))
        var m = nm.min(a)
        ```
    """

    if a.ndim == 1:
        return extrema_1d[is_max=False](a)
    else:
        return extrema_1d[is_max=False](ravel(a))


def min[dtype: DType](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Find the min value of an array along an axis.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the min is performed.

    Returns:
        An array with reduced number of dimensions.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.arange[f32](0, 6).reshape(Shape(2, 3))
        var m = nm.min(a, axis=1)
        ```
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            String("Error in `min`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    return apply_along_axis_reduce[func1d=extrema_1d[is_max=False]](
        a=a, axis=normalized_axis
    )


# ===-----------------------------------------------------------------------===#
# Array reductions (min/max over axes)
# ===-----------------------------------------------------------------------===#
def minimum[
    dtype: DType = DType.float64
](s1: SIMD[dtype, 1], s2: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    """
    Minimum value of two SIMD values.

    Parameters:
         dtype: The element type.

    Args:
        s1: A SIMD Value.
        s2: A SIMD Value.

    Returns:
        The minimum of the two SIMD Values as a SIMD Value of `dtype`.
    """
    return builtin_min(s1, s2)


def maximum[
    dtype: DType = DType.float64
](s1: SIMD[dtype, 1], s2: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    """
    Maximum value of two SIMD values.

    Parameters:
         dtype: The element type.

    Args:
        s1: A SIMD Value.
        s2: A SIMD Value.

    Returns:
        The maximum of the two SIMD Values as a SIMD Value of `dtype`.
    """
    return builtin_max(s1, s2)


def minimum[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise minimum of two arrays.

    Parameters:
         dtype: The element type.

    Args:
        array1: An array.
        array2: An array.

    Returns:
        The element-wise minimum of the two arrays as a array of `dtype`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f32]("[1, 3, 2]")
        var b = nm.array[f32]("[2, 1, 4]")
        var m = nm.minimum(a, b)
        ```
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return builtin_min(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)


def maximum[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise maximum of two arrays.

    Parameters:
         dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Returns:
        The element-wise maximum of the two arrays as a array of `dtype`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f32]("[1, 3, 2]")
        var b = nm.array[f32]("[2, 1, 4]")
        var m = nm.maximum(a, b)
        ```
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return builtin_max(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)
