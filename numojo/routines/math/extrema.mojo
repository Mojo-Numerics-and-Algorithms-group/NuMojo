# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Extrema finding
"""

# ===-----------------------------------------------------------------------===#
# SECTIONS:
# 1. Find extrema in elements of a single array.
# 2. Element-wise between elements of two arrays.
#
# TODO:
# 1) Add support for axis parameter.
# 2) Currently, constrained is crashing mojo, so commented it out and added raise Error. Check later.
# 3) Relax constrained[] to let user get whatever output they want, but make a warning instead.
# ===-----------------------------------------------------------------------===#

from algorithm import vectorize
from builtin.math import max as builtin_max
from builtin.math import min as builtin_min
from collections.optional import Optional
from sys import simdwidthof

from numojo.core.matrix import Matrix
import numojo.core.matrix as matrix
from numojo.core.ndarray import NDArray
import numojo.core.utility as utility
from numojo.routines.sorting import binary_sort


# ===-----------------------------------------------------------------------===#
# Find extrema in elements of a single array.
# ===-----------------------------------------------------------------------===#


fn extrema_1d[
    dtype: DType, //, max: Bool
](a: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Finds the max or min value in the buffer.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is the backend function for `max` and `min`, with or without `axis`.

    Parameters:
        dtype: The element type.
        max: If True, find max value, otherwise find min value.

    Args:
        a: An array.

    Returns:
        A tuple of the max or min value and its index.
    """

    var value = a._buf.ptr[0]

    @parameter
    if max:
        for i in range(a.size):
            if (a._buf.ptr + i)[] > value:
                value = a._buf.ptr[i]
                index = i
    else:
        for i in range(a.size):
            if (a._buf.ptr + i)[] < value:
                value = a._buf.ptr[i]
                index = i

    return value


fn max[dtype: DType](a: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Finds the max value of an array.
    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The max value.
    """

    if a.ndim == 1:
        a_flattened = a
    else:
        a_flattened = ravel(a)

    return extrema_1d[max=True](a=a_flattened)


fn max[dtype: DType](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Finds the max value of an array along the axis.
    The number of dimension will be reduced by 1.
    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the max is performed.

    Returns:
        An array with reduced number of dimensions.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            String("Error in `mean`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    return utility.apply_func_on_array_with_dim_reduction[
        func = extrema_1d[max=True]
    ](a=a, axis=normalized_axis)


fn max[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find max item. It is first flattened before sorting.
    """

    var max_value: Scalar[dtype]
    max_value, _ = _max(A, 0, A.size - 1)

    return max_value


fn max[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Find max item along the given axis.
    """
    if axis == 1:
        var B = Matrix[dtype](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _max(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    0
                ],
            )
        return B^
    elif axis == 0:
        return transpose(max(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn _find_extrema_and_index[
    dtype: DType, //, max: Bool
](A: NDArray[dtype]) raises -> Tuple[Scalar[dtype], Scalar[DType.index]]:
    """
    Finds the max or min value in the buffer and its index (first occurrence).
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
        dtype: The element type.
        max: If True, find max value, otherwise find min value.

    Args:
        A: An array.

    Returns:
        A tuple of the max or min value and its index.
    """

    var index: Scalar[DType.index] = 0
    var value = A._buf.ptr[0]

    @parameter
    if max:
        for i in range(A.size):
            if (A._buf.ptr + i)[] > value:
                value = A._buf.ptr[i]
                index = i
    else:
        for i in range(A.size):
            if (A._buf.ptr + i)[] < value:
                value = A._buf.ptr[i]
                index = i

    return Tuple(value, index)


fn _max[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Tuple[
    Scalar[dtype], Scalar[DType.index]
]:
    """
    Auxiliary function that find the max value in a range of the buffer.
    Both ends are included.
    """
    if (end >= A.size) or (start >= A.size):
        raise Error(
            String(
                "Index out of boundary! start={}, end={}, matrix.size={}"
            ).format(start, end, A.size)
        )

    var max_index: Scalar[DType.index] = start
    var max_value = A._buf.ptr[start]

    for i in range(start, end + 1):
        if A._buf.ptr[i] > max_value:
            max_value = A._buf.ptr[i]
            max_index = i

    return (max_value, max_index)


fn min[
    dtype: DType
](array: NDArray[dtype], axis: Int = 0) raises -> NDArray[dtype]:
    """Minumums of array elements over a given axis.

    Args:
        array: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """
    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.shape[i])
    if axis > ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    var axis_size: Int = shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(shape[i])
            slices.append(Slice(0, shape[i]))
        else:
            slices.append(Slice(0, 0))

    slices[axis] = Slice(0, 1)

    var result: NDArray[dtype] = array[slices]
    for i in range(1, axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        var mask1 = less(arr_slice, result)
        var mask2 = greater(arr_slice, result)
        # Wherever result is greater than the new slice it is set to zero
        # Wherever arr_slice is less than the old result it is added to fill those zeros
        result = add(
            result * bool_to_numeric[dtype](mask2),
            arr_slice * bool_to_numeric[dtype](mask1),
        )

    return result


fn min[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find min item. It is first flattened before sorting.
    """

    var min_value: Scalar[dtype]
    min_value, _ = _min(A, 0, A.size - 1)

    return min_value


fn min[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Find min item along the given axis.
    """
    if axis == 1:
        var B = Matrix[dtype](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _min(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    0
                ],
            )
        return B^
    elif axis == 0:
        return transpose(min(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn _min[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Tuple[
    Scalar[dtype], Scalar[DType.index]
]:
    """
    Auxiliary function that find the min value in a range of the buffer.
    Both ends are included.
    """
    if (end >= A.size) or (start >= A.size):
        raise Error(
            String(
                "Index out of boundary! start={}, end={}, matrix.size={}"
            ).format(start, end, A.size)
        )

    var min_index: Scalar[DType.index] = start
    var min_value = A._buf.ptr[start]

    for i in range(start, end + 1):
        if A._buf.ptr[i] < min_value:
            min_value = A._buf.ptr[i]
            min_index = i

    return (min_value, min_index)


fn minT[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Minimum value of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    alias width = simdwidthof[dtype]()
    var min_value = NDArray[dtype](NDArrayShape(width))
    for i in range(width):
        min_value.__setitem__(i, array[0])

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        min_value._buf.ptr.store(
            0,
            builtin_min(
                min_value._buf.ptr.load[width=simd_width](0),
                array._buf.ptr.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_min, width](array.num_elements())

    var result: Scalar[dtype] = Scalar[dtype](min_value.load(0))
    for i in range(min_value.__len__()):
        if min_value.load(i) < result:
            result = min_value.load(i)

    return result


# this roughly seems to be just an alias for min in numpy
fn amin[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Minimum value of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An array.
    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    return minT[dtype](array)


# this roughly seems to be just an alias for max in numpy
fn amax[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Maximum value of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A array.
    Returns:
        The maximum of all of the member values of array as a SIMD Value of `dtype`.
    """
    return maxT[dtype](array)


fn mimimum[
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


fn maximum[
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


# ===-----------------------------------------------------------------------===#
# Element-wise between elements of two arrays.
# ===-----------------------------------------------------------------------===#


fn minimum[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element wise minimum of two arrays.

    Parameters:
         dtype: The element type.

    Args:
        array1: An array.
        array2: An array.
    Returns:
        The element wise minimum of the two arrays as a array of `dtype`.
    """
    var result: NDArray[dtype] = NDArray[dtype](array1.shape)

    alias width = simdwidthof[dtype]()
    if array1.shape != array2.shape:
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        result._buf.ptr.store(
            idx,
            builtin_min(
                array1._buf.ptr.load[width=simd_width](idx),
                array2._buf.ptr.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_min, width](array1.num_elements())
    return result


fn maximum[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element wise maximum of two arrays.

    Parameters:
         dtype: The element type.

    Args:
        array1: A array.
        array2: A array.
    Returns:
        The element wise maximum of the two arrays as a array of `dtype`.
    """

    var result: NDArray[dtype] = NDArray[dtype](array1.shape)
    alias width = simdwidthof[dtype]()
    if array1.shape != array2.shape:
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        result._buf.ptr.store(
            idx,
            builtin_max(
                array1._buf.ptr.load[width=simd_width](idx),
                array2._buf.ptr.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_max, width](array1.num_elements())
    return result
