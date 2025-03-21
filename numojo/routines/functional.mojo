# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Functional programming.
"""

from algorithm.functional import vectorize, parallelize
from memory import memcpy
from sys import simdwidthof

from numojo.core.flags import Flags
from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides

# ===----------------------------------------------------------------------=== #
# `apply_along_axis`
#
# This section are OVERLOADS for the function `apply_along_axis` that
# applies a function to 1-d arrays along given axis
# It execute `func1d(a: NDArray, *args, **kwargs)` where
# `func1d` operates on 1-D arrays and
# `a` is a 1-d array slice of the original array along given axis.
# ===----------------------------------------------------------------------=== #

# The following overloads of `apply_along_axis` are for the case when the
# dimension of the input array is reduced.


fn apply_along_axis[
    dtype: DType,
    func1d: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> Scalar[
        dtype_func
    ],
](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
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
    var res: NDArray[dtype]

    if a.ndim == 1:
        res = numojo.creation._0darray[dtype](0)
        (res._buf.ptr).init_pointee_copy(func1d[dtype](a))

    else:
        res = NDArray[dtype](a.shape._pop(axis=axis))

        @parameter
        fn parallelized_func(i: Int):
            try:
                (res._buf.ptr + i).init_pointee_copy(
                    func1d[dtype](iterator.ith(i))
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


fn apply_along_axis[
    dtype: DType,
    func1d: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> Scalar[
        DType.index
    ],
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.index]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
    The returned data type is DType.index.
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
    var res: NDArray[DType.index]

    if a.ndim == 1:
        res = numojo.creation._0darray[DType.index](0)
        (res._buf.ptr).init_pointee_copy(func1d[dtype](a))

    else:
        res = NDArray[DType.index](a.shape._pop(axis=axis))

        @parameter
        fn parallelized_func(i: Int):
            try:
                (res._buf.ptr + i).init_pointee_copy(
                    func1d[dtype](iterator.ith(i))
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


fn apply_along_axis[
    dtype: DType, //,
    returned_dtype: DType,
    func1d: fn[dtype_func: DType, //, returned_dtype_func: DType] (
        NDArray[dtype_func]
    ) raises -> Scalar[returned_dtype_func],
](a: NDArray[dtype], axis: Int) raises -> NDArray[returned_dtype]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
    When the array is 1-d, the returned array will be a 0-d array.
    The target data type of the returned NDArray is different from the input
    NDArray. This is a function ***overload***.

    Raises:
        Error when the array is 1-d.

    Parameters:
        dtype: The data type of the input NDArray elements.
        returned_dtype: The data type of the output NDArray elements.
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
        res = numojo.creation._0darray[returned_dtype](0)
        (res._buf.ptr).init_pointee_copy(func1d[returned_dtype](a))

    else:
        res = NDArray[returned_dtype](a.shape._pop(axis=axis))

        @parameter
        fn parallelized_func(i: Int):
            try:
                (res._buf.ptr + i).init_pointee_copy(
                    func1d[returned_dtype](iterator.ith(i))
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


# The following overloads of `apply_along_axis` are for the case when the
# dimension of the input array is not reduced.


fn apply_along_axis[
    dtype: DType, //,
    func1d: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> NDArray[
        dtype_func
    ],
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
    var res = NDArray[dtype](a.shape)

    if a.flags.C_CONTIGUOUS and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        @parameter
        fn parallelized_func_c(i: Int):
            try:
                var elements: NDArray[dtype] = func1d[dtype](iterator.ith(i))
                memcpy(
                    res._buf.ptr + i * elements.size,
                    elements._buf.ptr,
                    elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        fn parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.index]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                indices, elements = iterator.ith_with_offsets(i)

                var res_along_axis: NDArray[dtype] = func1d[dtype](elements)

                for j in range(a.shape[axis]):
                    (res._buf.ptr + Int(indices[j])).init_pointee_copy(
                        (res_along_axis._buf.ptr + j)[]
                    )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


fn apply_along_axis[
    dtype: DType,
    func1d: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> NDArray[
        DType.index
    ],
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.index]:
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
    var res = NDArray[DType.index](a.shape)

    if a.flags.C_CONTIGUOUS and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        @parameter
        fn parallelized_func_c(i: Int):
            try:
                var elements: NDArray[DType.index] = func1d[dtype](
                    iterator.ith(i)
                )
                memcpy(
                    res._buf.ptr + i * elements.size,
                    elements._buf.ptr,
                    elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        fn parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.index]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                indices, elements = iterator.ith_with_offsets(i)

                var res_along_axis: NDArray[DType.index] = func1d[dtype](
                    elements
                )

                for j in range(a.shape[axis]):
                    (res._buf.ptr + Int(indices[j])).init_pointee_copy(
                        (res_along_axis._buf.ptr + j)[]
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

"""
If a and b have the same shape and strides, the function will be applied
element-wise to the two arrays.

Else if a and b have the same shape and the strides are both 1 for axis -1 or 0
(C or F contiguous is not sufficient due to broadcasted views),
the function with be applied by axis -1 or axis 0.

Else, conduct item-wise calculation.

If a and b have different shape (including when b is scalar), 
conduct a broadcasting.
"""
