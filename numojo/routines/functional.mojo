# ===----------------------------------------------------------------------=== #
# NuMojo: Functional
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Functional (numojo.routines.functional).
========================================
Functional programming utilities for array operations.

Implements functional utilities for NDArray operations such as `apply_along_axis`,
allowing application of functions along array axes.

Exports
-------
- `apply_along_axis_reduce`: Apply a reducing function along an axis.
- `apply_along_axis_reduce_with_dtype`: Apply a reducing function with explicit return dtype.
- `apply_along_axis_reduce_to_int`: Apply a reducing function returning integers.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm.functional import vectorize
from std.memory import unsafe_memcpy
from std.sys import simd_width_of

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from max.algorithm import parallelize

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.layout import Flags, NDArrayShape, NDArrayStrides
from numojo.core.ndarray import NDArray
from numojo.routines.creation import _0darray, arange

# ===----------------------------------------------------------------------=== #
# Apply Along Axis
# ===----------------------------------------------------------------------=== #

# The following overloads of `apply_along_axis` are for the case when the
# dimension of the input array is reduced.


# def apply_along_axis[
#     dtype: DType,
#     func1d: def[dtype_func: DType](NDArray[dtype_func]) raises thin -> Scalar[
#         dtype_func
#     ],
# ](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
#     """
#     Applies a function to a NDArray by axis and reduce that dimension.
#     When the array is 1-d, the returned array will be a 0-d array.

#     Parameters:
#         dtype: The data type of the input NDArray elements.
#         func1d: The function to apply to the NDArray.

#     Args:
#         a: The NDArray to apply the function to.
#         axis: The axis to apply the function to.

#     Returns:
#         The NDArray with the function applied to the input NDArray by axis.
#     """

#     # The iterator along the axis
#     var iterator = a.iter_along_axis(axis=axis)
#     # The final output array will have 1 less dimension than the input array
#     var res: NDArray[dtype]

#     if a.ndim == 1:
#         res = _0darray[dtype](0)
#         (res._buf.ptr).init_pointee_copy(func1d[dtype](a))

#     else:
#         res = NDArray[dtype](a.shape.pop(axis=axis))

#         @parameter
#         def parallelized_func(i: Int):
#             try:
#                 (res._buf.ptr + i).init_pointee_copy(
#                     func1d[dtype](iterator.ith(i))
#                 )
#             except e:
#                 print("Error in parallelized_func", e)

#         parallelize[parallelized_func](a.size // a.shape[axis])

#     return res^


def apply_along_axis_reduce_to_int[
    dtype: DType,
    func1d: def[dtype_func: DType](
        NDArray[dtype_func]
    ) capturing raises -> Scalar[DType.int],
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
    The returned data type is DType.int.
    When the array is 1-d, the returned array will be a 0-d array.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func1d: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """

    # The iterator along the axis
    var iterator = a.iter_along_axis(axis=axis)
    # The final output array will have 1 less dimension than the input array
    var res: NDArray[DType.int]

    if a.ndim == 1:
        res = _0darray[DType.int](0)
        res.unsafe_set(0, func1d[dtype](a))

    else:
        res = NDArray[DType.int](a.shape.pop(axis=axis))

        @parameter
        def parallelized_func(i: Int):
            try:
                res.unsafe_set(i, func1d[dtype](iterator.ith(i)))
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


def apply_along_axis_reduce[
    dtype: DType,
    func1d: def[dtype_func: DType](
        NDArray[dtype_func]
    ) capturing raises -> Scalar[dtype_func],
](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
    When the array is 1-d, the returned array will be a 0-d array.
    The target data type of the returned NDArray is different from the input
    NDArray. This is a function ***overload***.

    Raises:
        Error when the array is 1-d.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func1d: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """
    # The iterator along the axis
    # The final output array will have 1 less dimension than the input array
    var res: NDArray[dtype]

    if a.ndim == 1:
        res = _0darray[dtype](0)
        res.unsafe_set(0, func1d[dtype](a))

    else:
        var new_shape = a.shape.pop(axis=axis)
        res = NDArray[dtype](new_shape)
        var iterator = a.iter_along_axis(axis=axis)

        # for i in range(a.size // a.shape[axis]):
        #     var ith = iterator.ith(i)
        #     var func_result = func1d[dtype](ith)
        #     res._buf.store(i, func_result)

        @parameter
        def parallelized_func(i: Int):
            try:
                res.unsafe_set(i, func1d[dtype](iterator.ith(i)))
            except e:
                print("Error in parallelized_func", e)
            # try:
            #     (res._buf.ptr + i).init_pointee_copy(
            #         func1d[dtype](iterator.ith(i))
            #     )
            # except e:
            #     print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


def apply_along_axis_reduce_with_dtype[
    dtype: DType,
    returned_dtype: DType,
    func1d: def[dtype_func: DType, returned_dtype_func: DType](
        NDArray[dtype_func]
    ) capturing raises -> Scalar[returned_dtype_func],
](a: NDArray[dtype], axis: Int) raises -> NDArray[returned_dtype]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
    When the array is 1-d, the returned array will be a 0-d array.
    The function returns a different dtype than the input NDArray.

    Parameters:
        dtype: The data type of the input NDArray elements.
        returned_dtype: The data type of the returned NDArray elements.
        func1d: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """

    # The iterator along the axis
    var iterator = a.iter_along_axis(axis=axis)
    # The final output array will have 1 less dimension than the input array
    var res: NDArray[returned_dtype]

    if a.ndim == 1:
        res = _0darray[returned_dtype](0)
        res.unsafe_set(0, func1d[dtype, returned_dtype](a))

    else:
        res = NDArray[returned_dtype](a.shape.pop(axis=axis))

        @parameter
        def parallelized_func(i: Int):
            try:
                res.unsafe_set(
                    i, func1d[dtype, returned_dtype](iterator.ith(i))
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


# The following overloads of `apply_along_axis` are for the case when the
# dimension of the input array is not reduced.


def apply_along_axis_preserve[
    dtype: DType,
    func1d: def[dtype_func: DType](
        NDArray[dtype_func]
    ) capturing raises -> NDArray[dtype_func],
](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Applies a function to a NDArray by axis without reducing that dimension.
    The resulting array will have the same shape as the input array.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func1d: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """

    # The iterator along the axis
    var iterator = a.iter_along_axis(axis=axis)
    # The final output array will have the same shape as the input array
    var result: NDArray[dtype] = NDArray[dtype](a.shape)

    if a.is_c_contiguous() and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        @parameter
        def parallelized_func_c(i: Int):
            try:
                var elements: NDArray[dtype] = func1d[dtype](iterator.ith(i))
                unsafe_memcpy(
                    dest=result.unsafe_ptr().unsafe_offset(i * elements.size),
                    src=elements.unsafe_ptr(),
                    count=elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        def parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.int]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                var indices_elements = iterator.ith_with_offsets(i)
                indices = indices_elements[0].copy()
                elements = indices_elements[1].copy()
                # indices, elements = iterator.ith_with_offsets(i)

                var res_along_axis: NDArray[dtype] = func1d[dtype](elements)

                for j in range(a.shape[axis]):
                    result.unsafe_set(
                        Int(indices[j]), res_along_axis.unsafe_get(j)
                    )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return result^


# The following overloads of `apply_along_axis` are for the case when the
# dimension of the input array is not reduced.
# The function is applied in-place to the input array.
# For example, `sort_inplace()`.


def apply_along_axis_inplace[
    dtype: DType,
    func1d: def[dtype_func: DType](
        mut NDArray[dtype_func]
    ) capturing raises -> None,
](mut a: NDArray[dtype], axis: Int) raises -> None:
    """
    Applies a function to a NDArray by axis without reducing that dimension.
    The function is applied in-place to the input array.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func1d: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.
    """

    # The iterator along the axis
    var iterator = a.iter_along_axis(axis=axis)

    if a.is_c_contiguous() and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        @parameter
        def parallelized_func_c(i: Int):
            try:
                var elements: NDArray[dtype] = iterator.ith(i)
                func1d[dtype](elements)
                unsafe_memcpy(
                    dest=a.unsafe_ptr().unsafe_offset(i * elements.size),
                    src=elements.unsafe_ptr(),
                    count=elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        def parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.int]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                var indices_elements = iterator.ith_with_offsets(i)
                indices = indices_elements[0].copy()
                elements = indices_elements[1].copy()

                func1d[dtype](elements)

                for j in range(a.shape[axis]):
                    a.unsafe_set(Int(indices[j]), elements.unsafe_get(j))
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return None


def apply_along_axis_indices[
    dtype: DType,
    func1d: def[dtype_func: DType](
        NDArray[dtype_func]
    ) capturing raises -> NDArray[DType.int],
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """
    Applies a function to a NDArray by axis without reducing that dimension.
    The resulting array will have the same shape as the input array.
    The resulting array is an index array.
    It can be used for, e.g., argsort.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func1d: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The index array with the function applied to the input array by axis.
    """

    # The iterator along the axis
    var iterator = a.iter_along_axis(axis=axis)
    # The final output array will have the same shape as the input array
    var res = NDArray[DType.int](a.shape)

    if a.is_c_contiguous() and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        @parameter
        def parallelized_func_c(i: Int):
            try:
                var elements: NDArray[DType.int] = func1d[dtype](
                    iterator.ith(i)
                )
                unsafe_memcpy(
                    dest=res.unsafe_ptr().unsafe_offset(i * elements.size),
                    src=elements.unsafe_ptr(),
                    count=elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        def parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.int]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                var indices_elements = iterator.ith_with_offsets(i)
                indices = indices_elements[0].copy()
                elements = indices_elements[1].copy()

                var res_along_axis: NDArray[DType.int] = func1d[dtype](elements)

                for j in range(a.shape[axis]):
                    res.unsafe_set(
                        Int(indices[j]), res_along_axis.unsafe_get(j)
                    )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


# ===----------------------------------------------------------------------=== #
# `vectorize`
#
# This section are OVERLOADS for the function `vectorize` that
# applies a function to scalars to arrays.
# It execute `func(a: Scalar, b: Scalar, *args, **kwargs)` where
# `func` operates on scalars.
# ===----------------------------------------------------------------------=== #

# """
# If a and b have the same shape and strides, the function will be applied
# element-wise to the two arrays.

# Else if a and b have the same shape and the strides are both 1 for axis -1 or 0
# (C or F contiguous is not sufficient due to broadcasted views),
# the function with be applied by axis -1 or axis 0.

# Else, conduct item-wise calculation.

# If a and b have different shape (including when b is scalar),
# conduct a broadcasting.
# """
