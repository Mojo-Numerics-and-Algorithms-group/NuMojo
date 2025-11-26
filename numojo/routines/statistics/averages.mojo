# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Averages and variances
"""
# ===----------------------------------------------------------------------=== #
# Averages and variances
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
import math as mt

from numojo.core.ndarray import NDArray
from numojo.core.own_data import OwnData
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
import numojo.core.utility as utility
from numojo.routines.logic.comparison import greater, less
from numojo.routines.manipulation import broadcast_to, _broadcast_back_to
from numojo.routines.math.arithmetic import add
from numojo.routines.math.sums import sum, cumsum
import numojo.routines.math.misc as misc
from numojo.routines.sorting import sort


fn mean_1d[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype]) raises -> Scalar[returned_dtype]:
    """
    Calculate the arithmetic average of all items in an array.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is the backend function for `mean`, with or without `axis`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A 1-d array.

    Returns:
        A scalar defaulting to float64.
    """

    return sum(a).cast[returned_dtype]() / a.size


fn mean[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype]) raises -> Scalar[returned_dtype]:
    """
    Calculate the arithmetic average of all items in the array.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: NDArray.

    Returns:
        A scalar defaulting to float64.
    """
    return mean_1d[returned_dtype](a)


fn mean[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], axis: Int) raises -> NDArray[returned_dtype]:
    """
    Mean of array elements over a given axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: NDArray.
        axis: The axis along which the mean is performed.

    Returns:
        An NDArray.
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

    return numojo.apply_along_axis[
        returned_dtype=returned_dtype, func1d=mean_1d
    ](a=a, axis=normalized_axis)


fn mean[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: Matrix[dtype, **_]) -> Scalar[returned_dtype]:
    """
    Calculate the arithmetic average of all items in the Matrix.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A matrix.

    Returns:
        A scalar of the returned data type.
    """

    return sum(a).cast[returned_dtype]() / a.size


fn mean[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: Matrix[dtype, **_], axis: Int) raises -> Matrix[returned_dtype, OwnData]:
    """
    Calculate the arithmetic average of a Matrix along the axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A matrix.
        axis: The axis along which the mean is performed.

    Returns:
        A matrix of the returned data type.
    """

    if axis == 0:
        return sum(a, axis=0).astype[returned_dtype]() / a.shape[0]
    elif axis == 1:
        return sum(a, axis=1).astype[returned_dtype]() / a.shape[1]
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn median_1d[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype]) raises -> Scalar[returned_dtype]:
    """
    Median value of all items an array.
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
         dtype: The element type.
         returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A 1-d array.

    Returns:
        The median of all of the member values of array as a SIMD Value of `dtype`.
    """

    var sorted_array = sort(a)

    if a.size % 2 == 1:
        return sorted_array.item(a.size // 2).cast[returned_dtype]()
    else:
        return (
            sorted_array.item(a.size // 2 - 1) + sorted_array.item(a.size // 2)
        ).cast[returned_dtype]() / 2


fn median[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype]) raises -> Scalar[returned_dtype]:
    """
    Median value of all items of an array.

    Parameters:
         dtype: The element type.
         returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A 1-d array.

    Returns:
        The median of all of the member values of array as a SIMD Value of `dtype`.
    """

    return median_1d[returned_dtype](a)


fn median[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], axis: Int) raises -> NDArray[returned_dtype]:
    """
    Returns median of the array elements along the given axis.

    Parameters:
         dtype: The element type.
         returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        axis: The axis along which the median is performed.

    Returns:
        Median of the array elements along the given axis.
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
    return numojo.apply_along_axis[
        returned_dtype=returned_dtype, func1d=median_1d
    ](a=a, axis=normalized_axis)


fn mode_1d[dtype: DType](a: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Returns mode of all items of an array.
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
        dtype: The element type.

    Args:
        a: An NDArray.

    Returns:
        The mode of all of the member values of array as a SIMD Value of `dtype`.
    """

    var sorted_array: NDArray[dtype] = sort(a)
    var max_count = 0
    var mode_value = sorted_array.item(0)
    var current_count = 1

    for i in range(1, a.size):
        if sorted_array[i] == sorted_array[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                mode_value = sorted_array.item(i - 1)
            current_count = 1

    if current_count > max_count:
        mode_value = sorted_array.item(a.size - 1)

    return mode_value


fn mode[dtype: DType](array: NDArray[dtype]) raises -> Scalar[dtype]:
    """Mode of all items of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The mode of all of the member values of array as a SIMD Value of `dtype`.
    """

    return mode_1d(ravel(array))


fn mode[dtype: DType](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Returns mode of the array elements along the given axis.

    Parameters:
        dtype: The element type.

    Args:
        a: An NDArray.
        axis: The axis along which the mode is performed.

    Returns:
        Mode of the array elements along the given axis.
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

    return numojo.apply_along_axis[func1d=mode_1d](a=a, axis=normalized_axis)


fn std[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: NDArray[dtype], ddof: Int = 0) raises -> Scalar[returned_dtype]:
    """
    Compute the standard deviation.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: An array.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(
            String("ddof {} should be smaller than size {}").format(
                ddof, A.size
            )
        )

    return variance[returned_dtype](A, ddof=ddof) ** 0.5


fn std[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: NDArray[dtype], axis: Int, ddof: Int = 0) raises -> NDArray[
    returned_dtype
]:
    """
    Computes the standard deviation along the axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: An array.
        axis: The axis along which the mean is performed.
        ddof: Delta degree of freedom.

    Returns:
        An array.

    Raises:
        Error: If the axis is out of bounds.
        Error: If ddof is not smaller than the size of the axis.
    """

    var normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis += A.ndim
    if (normalized_axis >= A.ndim) or (normalized_axis < 0):
        raise Error(String("Axis {} out of bounds!").format(axis))

    for i in range(A.ndim):
        if ddof >= A.shape[i]:
            raise Error(
                String(
                    "ddof ({}) should be smaller than size ({}) of axis ({})"
                ).format(ddof, A.shape[i], i)
            )

    return variance[returned_dtype](A, axis=normalized_axis, ddof=ddof) ** 0.5


fn std[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: Matrix[dtype, **_], ddof: Int = 0) raises -> Scalar[returned_dtype]:
    """
    Compute the standard deviation.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: Matrix.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(
            String("ddof {} should be smaller than size {}").format(
                ddof, A.size
            )
        )

    return variance[returned_dtype](A, ddof=ddof) ** 0.5


fn std[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: Matrix[dtype, **_], axis: Int, ddof: Int = 0) raises -> Matrix[
    returned_dtype
]:
    """
    Compute the standard deviation along axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: Matrix.
        axis: 0 or 1.
        ddof: Delta degree of freedom.
    """

    return variance[returned_dtype](A, axis, ddof=ddof) ** 0.5


fn variance[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: NDArray[dtype], ddof: Int = 0) raises -> Scalar[returned_dtype]:
    """
    Compute the variance.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: An array.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(
            String("ddof {} should be smaller than size {}").format(
                ddof, A.size
            )
        )

    return sum(
        (A.astype[returned_dtype]() - mean[returned_dtype](A))
        * (A.astype[returned_dtype]() - mean[returned_dtype](A))
    ) / (A.size - ddof)


fn variance[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: NDArray[dtype], axis: Int, ddof: Int = 0) raises -> NDArray[
    returned_dtype
]:
    """
    Computes the variance along the axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: An array.
        axis: The axis along which the mean is performed.
        ddof: Delta degree of freedom.

    Returns:
        An array.

    Raises:
        Error: If the axis is out of bounds.
        Error: If ddof is not smaller than the size of the axis.
    """

    var normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis += A.ndim
    if (normalized_axis >= A.ndim) or (normalized_axis < 0):
        raise Error(String("Axis {} out of bounds!").format(axis))

    for i in range(A.ndim):
        if ddof >= A.shape[i]:
            raise Error(
                String(
                    "ddof ({}) should be smaller than size ({}) of axis ({})"
                ).format(ddof, A.shape[i], i)
            )

    return sum(
        (
            A.astype[returned_dtype]()
            - _broadcast_back_to(
                mean[returned_dtype](A, axis=normalized_axis),
                A.shape,
                axis=normalized_axis,
            )
        )
        * (
            A.astype[returned_dtype]()
            - _broadcast_back_to(
                mean[returned_dtype](A, axis=normalized_axis),
                A.shape,
                axis=normalized_axis,
            )
        ),
        axis=normalized_axis,
    ) / (A.shape[normalized_axis] - ddof)


fn variance[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: Matrix[dtype, **_], ddof: Int = 0) raises -> Scalar[returned_dtype]:
    """
    Compute the variance.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: Matrix.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(
            String("ddof {} should be smaller than size {}").format(
                ddof, A.size
            )
        )

    return sum(
        (A.astype[returned_dtype]() - mean[returned_dtype](A))
        * (A.astype[returned_dtype]() - mean[returned_dtype](A))
    ) / (A.size - ddof)


fn variance[
    dtype: DType, //, returned_dtype: DType = DType.float64
](A: Matrix[dtype, **_], axis: Int, ddof: Int = 0) raises -> Matrix[
    returned_dtype
]:
    """
    Compute the variance along axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        A: Matrix.
        axis: 0 or 1.
        ddof: Delta degree of freedom.
    """

    if (ddof >= A.shape[0]) or (ddof >= A.shape[1]):
        raise Error(
            String("ddof {} should be smaller than size {}x{}").format(
                ddof, A.shape[0], A.shape[1]
            )
        )

    if axis == 0:
        return sum(
            (A.astype[returned_dtype]() - mean[returned_dtype](A, axis=0))
            * (A.astype[returned_dtype]() - mean[returned_dtype](A, axis=0)),
            axis=0,
        ) / (A.shape[0] - ddof)
    elif axis == 1:
        return sum(
            (A.astype[returned_dtype]() - mean[returned_dtype](A, axis=1))
            * (A.astype[returned_dtype]() - mean[returned_dtype](A, axis=1)),
            axis=1,
        ) / (A.shape[1] - ddof)
    else:
        raise Error(String("The axis can either be 1 or 0!"))
