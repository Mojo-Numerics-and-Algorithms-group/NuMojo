# ===----------------------------------------------------------------------=== #
# NuMojo: Extrema routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Extrema routines for NuMojo (numojo.routines.math.extrema).

Contains min/max helpers for NDArrays and Matrices, including axis-aware reductions
and element-wise comparisons.
"""

from std.algorithm import vectorize
from max.algorithm import parallelize
import std.math.math as stdlib_math
from std.math import max as builtin_max
from std.math import min as builtin_min
from std.collections.optional import Optional
from std.sys import simd_width_of

from numojo.core.matrix import Matrix
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
    var value = a._buf.load[width=1](0)

    comptime if is_max:

        def vectorize_max[simd_width: Int](offset: Int) {mut value, a}:
            var temp = a._buf.ptr.unsafe_load[width=simd_width](
                offset
            ).reduce_max()
            if temp >= value:
                value = temp

        vectorize[simd_width](a.size, vectorize_max)

        return value

    else:

        def vectorize_min[simd_width: Int](offset: Int) {mut value, a} -> None:
            var temp = a._buf.ptr.unsafe_load[width=simd_width](
                offset
            ).reduce_min()
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
# Matrix reductions (min/max over axes)
# ===-----------------------------------------------------------------------===#


@always_inline
def matrix_extrema[
    dtype: DType, find_max: Bool
](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Generic implementation for finding global min/max in a matrix.

    Works with any memory layout (row-major or column-major).
    """
    var extreme_val = A[0, 0]

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            var current = A[i, j]
            if find_max:
                if current > extreme_val:
                    extreme_val = current
            else:
                if current < extreme_val:
                    extreme_val = current

    return extreme_val


@always_inline
def matrix_extrema_axis[
    dtype: DType, find_max: Bool
](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Generic implementation for finding min/max along an axis in a matrix.

    Works with any memory layout (row-major or column-major).
    """
    if axis != 0 and axis != 1:
        raise Error(String("The axis can either be 1 or 0!"))

    var B = Matrix[dtype](
        shape=(A.shape[0], 1) if axis == 1 else (1, A.shape[1])
    )

    if axis == 1:
        for i in range(A.shape[0]):
            var extreme_val = A[i, 0]

            for j in range(1, A.shape[1]):
                var current = A[i, j]

                if find_max:
                    if current > extreme_val:
                        extreme_val = current
                else:
                    if current < extreme_val:
                        extreme_val = current

            B[i, 0] = extreme_val
    else:
        for j in range(A.shape[1]):
            var extreme_val = A[0, j]

            for i in range(1, A.shape[0]):
                var current = A[i, j]

                if find_max:
                    if current > extreme_val:
                        extreme_val = current
                else:
                    if current < extreme_val:
                        extreme_val = current

            B[0, j] = extreme_val

    return B^


def max[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find max item.

    Args:
        A: A Matrix.

    Returns:
        The max value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var A = Matrix.rand[f32](shape=(3, 3))
        var m = nm.max(A)
        ```
    """
    return matrix_extrema[dtype, True](A)


def max[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Find max item along the given axis.

    Args:
        A: A Matrix.
        axis: The axis along which the max is performed.

    Returns:
        A Matrix with reduced dimensions along the axis.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var A = Matrix.rand[f32](shape=(3, 3))
        var m = nm.max(A, axis=1)
        ```
    """
    return matrix_extrema_axis[dtype, True](A, axis)


def _max[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Tuple[
    Scalar[dtype], Scalar[DType.int]
]:
    """
    Auxiliary function that finds the max value in a range of the buffer.

    Both ends are included.
    """
    if (end >= A.size) or (start >= A.size):
        raise Error(
            String(
                "Index out of boundary! start={}, end={}, matrix.size={}"
            ).format(start, end, A.size)
        )

    var max_index: Int = start

    var rows = A.shape[0]
    var cols = A.shape[1]

    var start_row: Int
    var start_col: Int

    if A.is_f_contiguous():
        start_col = start // rows
        start_row = start % rows
    else:
        start_row = start // cols
        start_col = start % cols

    var max_value = A[start_row, start_col]

    for i in range(start, end + 1):
        var row: Int
        var col: Int

        if A.is_f_contiguous():
            col = i // rows
            row = i % rows
        else:
            row = i // cols
            col = i % cols

        if row < rows and col < cols:
            var current_value = A[row, col]
            if current_value > max_value:
                max_value = current_value
                max_index = i

    return (max_value, Scalar[DType.int](max_index))


def min[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find min item.

    Args:
        A: A Matrix.

    Returns:
        The min value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var A = Matrix.rand[f32](shape=(3, 3))
        var m = nm.min(A)
        ```
    """
    return matrix_extrema[dtype, False](A)


def min[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Find min item along the given axis.

    Args:
        A: A Matrix.
        axis: The axis along which the min is performed.

    Returns:
        A Matrix with reduced dimensions along the axis.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var A = Matrix.rand[f32](shape=(3, 3))
        var m = nm.min(A, axis=0)
        ```
    """
    return matrix_extrema_axis[dtype, False](A, axis)


def _min[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Tuple[
    Scalar[dtype], Scalar[DType.int]
]:
    """
    Auxiliary function that finds the min value in a range of the buffer.

    Both ends are included.
    """
    if (end >= A.size) or (start >= A.size):
        raise Error(
            String(
                "Index out of boundary! start={}, end={}, matrix.size={}"
            ).format(start, end, A.size)
        )

    var min_index: Int = start

    var rows = A.shape[0]
    var cols = A.shape[1]

    var start_row: Int
    var start_col: Int

    if A.is_f_contiguous():
        start_col = start // rows
        start_row = start % rows
    else:
        start_row = start // cols
        start_col = start % cols

    var min_value = A[start_row, start_col]

    for i in range(start, end + 1):
        var row: Int
        var col: Int

        if A.is_f_contiguous():
            col = i // rows
            row = i % rows
        else:
            row = i // cols
            col = i % cols

        if row < rows and col < cols:
            var current_value = A[row, col]
            if current_value < min_value:
                min_value = current_value
                min_index = i

    return (min_value, Scalar[DType.int](min_index))


# ===-----------------------------------------------------------------------===#
# Element-wise pairwise extrema
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
