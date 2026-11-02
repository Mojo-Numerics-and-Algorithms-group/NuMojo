# ===----------------------------------------------------------------------=== #
# NuMojo: NaN-aware statistics routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
NaN-aware statistics (numojo.routines.statistics.nanfunctions).
===============================================================
Reductions that treat `NaN` elements as missing values.

Mirrors `numpy`'s `nan*` family: each function behaves like its
non-`nan`-prefixed counterpart, except that `NaN` elements are excluded
from the computation instead of propagating into the result.

Exports
-------
- `nansum`: Sum of array elements, ignoring `NaN`.
- `nanmean`: Arithmetic mean of array elements, ignoring `NaN`.
- `nanmax`: Maximum array element, ignoring `NaN`.
- `nanmin`: Minimum array element, ignoring `NaN`.
- `nanvar`: Variance of array elements, ignoring `NaN`.
- `nanstd`: Standard deviation of array elements, ignoring `NaN`.
- `nanmedian`: Median value of array elements, ignoring `NaN`.
- `nanargmax`: Index of the maximum array element, ignoring `NaN`.
- `nanargmin`: Index of the minimum array element, ignoring `NaN`.
- `nanprod`: Product of array elements, treating `NaN` as one.
- `nancumsum`: Cumulative sum of array elements, treating `NaN` as zero.
- `nancumprod`: Cumulative product of array elements, treating `NaN` as
  one.
- `nanpercentile`: Percentile of array elements, ignoring `NaN`.
- `nanquantile`: Quantile of array elements, ignoring `NaN`.
- `nan_to_num`: Replaces `NaN` and infinite elements with finite values.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math
from std.utils.numerics import max_finite, min_finite

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import _0darray
from numojo.routines.functional import (
    apply_along_axis_reduce,
    apply_along_axis_reduce_to_int,
)
from numojo.routines.manipulation import ravel
from numojo.routines.math.products import cumprod, prod
from numojo.routines.math.sums import cumsum
from numojo.routines.sorting import sort
from numojo.routines.statistics.percentiles import quantile_1d


def nansum_1d[
    dtype: DType, //
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """
    Sum all items in an array, treating `NaN` elements as zero.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is the backend function for `nansum`, with or without `axis`.

    Parameters:
        dtype: The element type.

    Args:
        a: A 1-d array.

    Returns:
        The sum of the non-`NaN` elements as a scalar of `dtype`.
    """

    var total = Scalar[dtype](0)
    for i in range(a.size):
        var value = a.item(i)
        if not math.isnan(value):
            total += value
    return total


def nansum[
    dtype: DType
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Sum of all items in the array, treating `NaN` elements as zero.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.

    Returns:
        The sum of the non-`NaN` elements as a scalar of `dtype`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nansum(a))  # 4.0
        ```
    """
    return nansum_1d(ravel(a))


def nansum[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Sum of array elements over a given axis, treating `NaN` elements as
    zero.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nansum`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nansum",
            )
        )

    return apply_along_axis_reduce[dtype, func1d=nansum_1d](
        a=a, axis=normalized_axis
    )


def nanmean_1d[
    dtype: DType, //
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """
    Calculate the arithmetic mean of all items in an array, ignoring `NaN`
    elements. Regardless of the shape of input, it is treated as a 1-d
    array. It is the backend function for `nanmean`, with or without
    `axis`.

    Parameters:
        dtype: The element type.

    Args:
        a: A 1-d array.

    Returns:
        The mean of the non-`NaN` elements as a scalar of `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.
    """

    var total = Scalar[dtype](0)
    var count = 0
    for i in range(a.size):
        var value = a.item(i)
        if not math.isnan(value):
            total += value
            count += 1

    if count == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Error in `nanmean`: All-NaN slice encountered.",
                location="nanmean",
            )
        )

    return total / Scalar[dtype](count)


def nanmean[
    dtype: DType
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Calculate the arithmetic mean of all items in the array, ignoring
    `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.

    Returns:
        The mean of the non-`NaN` elements as a scalar of `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanmean(a))  # 2.0
        ```
    """
    return nanmean_1d(ravel(a))


def nanmean[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Mean of array elements over a given axis, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the mean is performed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanmean`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanmean",
            )
        )

    return apply_along_axis_reduce[dtype, func1d=nanmean_1d](
        a=a, axis=normalized_axis
    )


def nanextrema_1d[
    dtype: DType, //, is_max: Bool
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """
    Find the max or min value in the buffer, ignoring `NaN` elements.

    The input is treated as a 1-D array regardless of shape. This is the
    backend routine for `nanmax` and `nanmin`.

    Parameters:
        dtype: The element type.
        is_max: If True, find max value, otherwise find min value.

    Args:
        a: An array.

    Returns:
        The extreme non-`NaN` value.

    Raises:
        NumojoError: If all elements are `NaN`.
    """

    var found = False
    var value = Scalar[dtype](0)
    for i in range(a.size):
        var candidate = a.item(i)
        if math.isnan(candidate):
            continue
        if not found:
            value = candidate
            found = True
        elif is_max:
            if candidate > value:
                value = candidate
        else:
            if candidate < value:
                value = candidate

    if not found:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Error in `{}`: All-NaN slice encountered."
                ).format("nanmax" if is_max else "nanmin"),
                location="nanmax" if is_max else "nanmin",
            )
        )

    return value


def nanextrema_1d_max[
    dtype: DType, //
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """Find the max value in a 1-D array, ignoring `NaN` elements."""
    return nanextrema_1d[is_max=True](a)


def nanextrema_1d_min[
    dtype: DType, //
](a: NDArray[dtype]) capturing raises -> Scalar[dtype]:
    """Find the min value in a 1-D array, ignoring `NaN` elements."""
    return nanextrema_1d[is_max=False](a)


def nanmax[
    dtype: DType
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Find the max value of an array, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The max non-`NaN` value.

    Raises:
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanmax(a))  # 3.0
        ```
    """
    return nanextrema_1d[is_max=True](ravel(a))


def nanmax[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Find the max value of an array along an axis, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the max is performed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanmax`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanmax",
            )
        )

    return apply_along_axis_reduce[dtype, func1d=nanextrema_1d_max](
        a=a, axis=normalized_axis
    )


def nanmin[
    dtype: DType
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Find the min value of an array, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The min non-`NaN` value.

    Raises:
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanmin(a))  # 1.0
        ```
    """
    return nanextrema_1d[is_max=False](ravel(a))


def nanmin[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Find the min value of an array along an axis, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the min is performed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanmin`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanmin",
            )
        )

    return apply_along_axis_reduce[dtype, func1d=nanextrema_1d_min](
        a=a, axis=normalized_axis
    )


def nanvar_1d[
    dtype: DType, //
](a: NDArray[dtype], ddof: Int = 0) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    Compute the variance of all items in a 1-d array, ignoring `NaN`
    elements. It is the backend function for `nanvar`, with or without
    `axis`.

    Parameters:
        dtype: The element type.

    Args:
        a: A 1-d array.
        ddof: Delta degree of freedom.

    Returns:
        The variance of the non-`NaN` elements as a scalar of `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.
        NumojoError: If `ddof` is not smaller than the number of non-`NaN`
            elements.
    """

    var total = Scalar[dtype](0)
    var count = 0
    for i in range(a.size):
        var value = a.item(i)
        if not math.isnan(value):
            total += value
            count += 1

    if count == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Error in `nanvar`: All-NaN slice encountered.",
                location="nanvar",
            )
        )
    if ddof >= count:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Error in `nanvar`: ddof {} should be smaller than the"
                    " number of non-NaN elements {}"
                ).format(ddof, count),
                location="nanvar",
            )
        )

    var mean_value = total / Scalar[dtype](count)

    var sq_total = Scalar[dtype](0)
    for i in range(a.size):
        var value = a.item(i)
        if not math.isnan(value):
            var deviation = value - mean_value
            sq_total += deviation * deviation

    return sq_total / Scalar[dtype](count - ddof)


def nanvar[
    dtype: DType
](a: NDArray[dtype], ddof: Int = 0) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    Compute the variance of all items in the array, ignoring `NaN`
    elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        ddof: Delta degree of freedom.

    Returns:
        The variance of the non-`NaN` elements as a scalar of `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.
        NumojoError: If `ddof` is not smaller than the number of non-`NaN`
            elements.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanvar(a))  # 1.0
        ```
    """
    return nanvar_1d(ravel(a), ddof=ddof)


def nanvar[
    dtype: DType
](a: NDArray[dtype], axis: Int, ddof: Int = 0) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Variance of array elements over a given axis, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the variance is performed.
        ddof: Delta degree of freedom.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
        NumojoError: If `ddof` is not smaller than the number of non-`NaN`
            elements in a slice.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanvar`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanvar",
            )
        )

    if a.ndim == 1:
        return _0darray[dtype](nanvar_1d(a, ddof=ddof))

    var new_shape = a.shape.pop(axis=normalized_axis)
    var res = NDArray[dtype](new_shape)
    var iterator = a.iter_along_axis(axis=normalized_axis)
    for i in range(a.size // a.shape[normalized_axis]):
        res.unsafe_set(i, nanvar_1d(iterator.ith(i), ddof=ddof))

    return res^


def nanstd[
    dtype: DType
](a: NDArray[dtype], ddof: Int = 0) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    Compute the standard deviation of all items in the array, ignoring
    `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        ddof: Delta degree of freedom.

    Returns:
        The standard deviation of the non-`NaN` elements as a scalar of
        `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.
        NumojoError: If `ddof` is not smaller than the number of non-`NaN`
            elements.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanstd(a))  # 1.0
        ```
    """
    return nanvar(a, ddof=ddof) ** Scalar[dtype](0.5)


def nanstd[
    dtype: DType
](a: NDArray[dtype], axis: Int, ddof: Int = 0) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Standard deviation of array elements over a given axis, ignoring `NaN`
    elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the standard deviation is performed.
        ddof: Delta degree of freedom.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
        NumojoError: If `ddof` is not smaller than the number of non-`NaN`
            elements in a slice.
    """
    return nanvar(a, axis=axis, ddof=ddof) ** Scalar[dtype](0.5)


def nanmedian_1d[
    dtype: DType, //
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the median of all items in a 1-d array, ignoring `NaN`
    elements. It is the backend function for `nanmedian`, with or without
    `axis`.

    Parameters:
        dtype: The element type.

    Args:
        a: A 1-d array.

    Returns:
        The median of the non-`NaN` elements as a scalar of `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.
    """

    var count = 0
    for i in range(a.size):
        if not math.isnan(a.item(i)):
            count += 1

    if count == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Error in `nanmedian`: All-NaN slice encountered.",
                location="nanmedian",
            )
        )

    var filtered = NDArray[dtype](Shape(count))
    var idx = 0
    for i in range(a.size):
        var value = a.item(i)
        if not math.isnan(value):
            filtered.itemset(idx, value)
            idx += 1

    var sorted_array = sort(filtered)
    if count % 2 == 1:
        return sorted_array.item(count // 2)
    else:
        return (
            sorted_array.item(count // 2 - 1) + sorted_array.item(count // 2)
        ) / 2


def nanmedian[
    dtype: DType
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the median of all items in the array, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.

    Returns:
        The median of the non-`NaN` elements as a scalar of `dtype`.

    Raises:
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanmedian(a))  # 2.0
        ```
    """
    return nanmedian_1d(ravel(a))


def nanmedian[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Median of array elements over a given axis, ignoring `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the median is performed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanmedian`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanmedian",
            )
        )

    if a.ndim == 1:
        return _0darray[dtype](nanmedian_1d(a))

    var new_shape = a.shape.pop(axis=normalized_axis)
    var res = NDArray[dtype](new_shape)
    var iterator = a.iter_along_axis(axis=normalized_axis)
    for i in range(a.size // a.shape[normalized_axis]):
        res.unsafe_set(i, nanmedian_1d(iterator.ith(i)))

    return res^


def nanargmax_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[DType.int]:
    """
    Returns the index of the maximum value in the buffer, ignoring `NaN`
    elements. Regardless of the shape of input, it is treated as a 1-d
    array. It is the backend function for `nanargmax`, with or without
    `axis`.

    Parameters:
        dtype: The element type.

    Args:
        a: A 1-d array.

    Returns:
        The index of the maximum non-`NaN` value in the buffer.

    Raises:
        NumojoError: If all elements are `NaN`.
    """

    var found = False
    var value = Scalar[dtype](0)
    var result = 0
    for i in range(a.size):
        var candidate = a.item(i)
        if math.isnan(candidate):
            continue
        if not found:
            value = candidate
            result = i
            found = True
        elif candidate > value:
            value = candidate
            result = i

    if not found:
        raise Error(
            NumojoError(
                category="value",
                message="Error in `nanargmax`: All-NaN slice encountered.",
                location="nanargmax",
            )
        )

    return Scalar[DType.int](result)


def nanargmin_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[DType.int]:
    """
    Returns the index of the minimum value in the buffer, ignoring `NaN`
    elements. Regardless of the shape of input, it is treated as a 1-d
    array. It is the backend function for `nanargmin`, with or without
    `axis`.

    Parameters:
        dtype: The element type.

    Args:
        a: A 1-d array.

    Returns:
        The index of the minimum non-`NaN` value in the buffer.

    Raises:
        NumojoError: If all elements are `NaN`.
    """

    var found = False
    var value = Scalar[dtype](0)
    var result = 0
    for i in range(a.size):
        var candidate = a.item(i)
        if math.isnan(candidate):
            continue
        if not found:
            value = candidate
            result = i
            found = True
        elif candidate < value:
            value = candidate
            result = i

    if not found:
        raise Error(
            NumojoError(
                category="value",
                message="Error in `nanargmin`: All-NaN slice encountered.",
                location="nanargmin",
            )
        )

    return Scalar[DType.int](result)


def nanargmax[
    dtype: DType, //
](a: NDArray[dtype]) raises -> Scalar[
    DType.int
] where dtype.is_floating_point():
    """
    Returns the index of the maximum value of the array, ignoring `NaN`
    elements. The array is flattened before the index is computed.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The index of the maximum non-`NaN` value in the flattened array.

    Raises:
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanargmax(a))  # 2
        ```
    """
    return nanargmax_1d(ravel(a))


def nanargmax[
    dtype: DType, //
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    DType.int
] where dtype.is_floating_point():
    """
    Index of the maximum value of the array along a given axis, ignoring
    `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the index is computed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanargmax`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanargmax",
            )
        )

    return apply_along_axis_reduce_to_int[dtype, func1d=nanargmax_1d](
        a=a, axis=normalized_axis
    )


def nanargmin[
    dtype: DType, //
](a: NDArray[dtype]) raises -> Scalar[
    DType.int
] where dtype.is_floating_point():
    """
    Returns the index of the minimum value of the array, ignoring `NaN`
    elements. The array is flattened before the index is computed.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The index of the minimum non-`NaN` value in the flattened array.

    Raises:
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanargmin(a))  # 0
        ```
    """
    return nanargmin_1d(ravel(a))


def nanargmin[
    dtype: DType, //
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    DType.int
] where dtype.is_floating_point():
    """
    Index of the minimum value of the array along a given axis, ignoring
    `NaN` elements.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which the index is computed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanargmin`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanargmin",
            )
        )

    return apply_along_axis_reduce_to_int[dtype, func1d=nanargmin_1d](
        a=a, axis=normalized_axis
    )


def _nan_fill[
    dtype: DType, //
](a: NDArray[dtype], fill: Scalar[dtype]) raises -> NDArray[dtype]:
    """Internal: returns a copy of `a` with every `NaN` element replaced
    by `fill`."""
    var result = NDArray[dtype](a.shape)
    for i in range(a.size):
        var value = a.item(i)
        result.itemset(i, fill if math.isnan(value) else value)
    return result^


def nanprod[
    dtype: DType
](a: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Product of all items in the array, treating `NaN` elements as one.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.

    Returns:
        The product of the array, with `NaN` elements treated as one.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanprod(a))  # 3.0
        ```
    """
    return prod(_nan_fill(a, Scalar[dtype](1)))


def nanprod[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Product of array elements over a given axis, treating `NaN` elements
    as one.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the product is performed.

    Returns:
        An array with reduced number of dimensions.

    Raises:
        NumojoError: If the axis is out of bound.
    """
    return prod(_nan_fill(a, Scalar[dtype](1)), axis)


def nancumsum[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Cumulative sum of all items of the array, treating `NaN` elements as
    zero. The array is flattened before the cumulative sum is computed.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.

    Returns:
        The cumulative sum of the flattened array, with `NaN` elements
        treated as zero.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nancumsum(a))  # [1.0, 1.0, 4.0]
        ```
    """
    return cumsum(_nan_fill(a, Scalar[dtype](0)))


def nancumsum[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Cumulative sum of array elements over a given axis, treating `NaN`
    elements as zero.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the cumulative sum is performed.

    Returns:
        An array with the same shape as `a`.

    Raises:
        NumojoError: If the axis is out of bound.
    """
    return cumsum(_nan_fill(a, Scalar[dtype](0)), axis)


def nancumprod[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Cumulative product of all items of the array, treating `NaN` elements
    as one. The array is flattened before the cumulative product is
    computed.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.

    Returns:
        The cumulative product of the flattened array, with `NaN`
        elements treated as one.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nancumprod(a))  # [1.0, 1.0, 3.0]
        ```
    """
    return cumprod(_nan_fill(a, Scalar[dtype](1)))


def nancumprod[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Cumulative product of array elements over a given axis, treating
    `NaN` elements as one.

    Parameters:
        dtype: The element type.

    Args:
        a: NDArray.
        axis: The axis along which the cumulative product is performed.

    Returns:
        An array with the same shape as `a`.

    Raises:
        NumojoError: If the axis is out of bound.
    """
    return cumprod(_nan_fill(a, Scalar[dtype](1)), axis)


def nanquantile_1d[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], q: Scalar[returned_dtype]) raises -> Scalar[
    returned_dtype
]:
    """
    Quantile of all items of a 1-d array using linear interpolation,
    ignoring `NaN` elements. It is the backend function for
    `nanquantile`, with or without `axis`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A 1-d array.
        q: Quantile to compute, in the inclusive range `[0, 1]`.

    Returns:
        The `q`-th quantile of the non-`NaN` elements as a scalar of
        `returned_dtype`.

    Raises:
        NumojoError: If `q` is not in the range `[0, 1]`.
        NumojoError: If all elements are `NaN`.
    """

    var count = 0
    for i in range(a.size):
        if not math.isnan(a.item(i)):
            count += 1

    if count == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Error in `nanquantile`: All-NaN slice encountered.",
                location="nanquantile",
            )
        )

    var filtered = NDArray[dtype](Shape(count))
    var idx = 0
    for i in range(a.size):
        var value = a.item(i)
        if not math.isnan(value):
            filtered.itemset(idx, value)
            idx += 1

    return quantile_1d[returned_dtype](filtered, q)


def nanquantile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], q: Scalar[returned_dtype]) raises -> Scalar[
    returned_dtype
]:
    """
    Quantile of all items of the array using linear interpolation,
    ignoring `NaN` elements.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        q: Quantile to compute, in the inclusive range `[0, 1]`.

    Returns:
        The `q`-th quantile of the non-`NaN` elements as a scalar of
        `returned_dtype`.

    Raises:
        NumojoError: If `q` is not in the range `[0, 1]`.
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanquantile(a, Float64(0.5)))  # 2.0
        ```
    """
    return nanquantile_1d[returned_dtype](ravel(a), q)


def nanquantile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], q: Scalar[returned_dtype], axis: Int) raises -> NDArray[
    returned_dtype
]:
    """
    Quantile of the array elements along the given axis, ignoring `NaN`
    elements.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        q: Quantile to compute, in the inclusive range `[0, 1]`.
        axis: The axis along which the quantile is computed.

    Returns:
        An array of quantiles with `axis` removed.

    Raises:
        NumojoError: If `axis` is out of bounds.
        NumojoError: If `q` is not in the range `[0, 1]`.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    var normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `nanquantile`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="nanquantile",
            )
        )

    if a.ndim == 1:
        return _0darray[returned_dtype](nanquantile_1d[returned_dtype](a, q))

    var res = NDArray[returned_dtype](a.shape.pop(axis=normalized_axis))
    var iterator = a.iter_along_axis(axis=normalized_axis)
    for i in range(a.size // a.shape[normalized_axis]):
        res.unsafe_set(i, nanquantile_1d[returned_dtype](iterator.ith(i), q))

    return res^


def nanpercentile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], p: Scalar[returned_dtype]) raises -> Scalar[
    returned_dtype
]:
    """
    Percentile of all items of the array using linear interpolation,
    ignoring `NaN` elements.

    Equivalent to `nanquantile(a, p / 100)`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        p: Percentile to compute, in the inclusive range `[0, 100]`.

    Returns:
        The `p`-th percentile of the non-`NaN` elements as a scalar of
        `returned_dtype`.

    Raises:
        NumojoError: If `p` is not in the range `[0, 100]`.
        NumojoError: If all elements are `NaN`.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, 3.0]")
        print(nm.nanpercentile(a, Float64(50)))  # 2.0
        ```
    """

    if (p < 0) or (p > 100):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "`p` must be in the range [0, 100], got {}"
                ).format(p),
                location="nanpercentile",
            )
        )

    return nanquantile_1d[returned_dtype](ravel(a), p / 100)


def nanpercentile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], p: Scalar[returned_dtype], axis: Int) raises -> NDArray[
    returned_dtype
]:
    """
    Percentile of the array elements along the given axis, ignoring `NaN`
    elements.

    Equivalent to `nanquantile(a, p / 100, axis=axis)`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        p: Percentile to compute, in the inclusive range `[0, 100]`.
        axis: The axis along which the percentile is computed.

    Returns:
        An array of percentiles with `axis` removed.

    Raises:
        NumojoError: If `axis` is out of bounds.
        NumojoError: If `p` is not in the range `[0, 100]`.
        NumojoError: If a slice along the axis is all `NaN`.
    """

    if (p < 0) or (p > 100):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "`p` must be in the range [0, 100], got {}"
                ).format(p),
                location="nanpercentile",
            )
        )

    return nanquantile[returned_dtype](a, p / 100, axis)


def nan_to_num[
    dtype: DType, //
](
    a: NDArray[dtype],
    nan: Scalar[dtype] = 0,
    posinf: Optional[Scalar[dtype]] = None,
    neginf: Optional[Scalar[dtype]] = None,
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Replaces `NaN` elements with `nan`, positive infinity with `posinf`,
    and negative infinity with `neginf`.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        nan: The value used to replace `NaN` elements. Defaults to 0.
        posinf: The value used to replace positive infinity. Defaults to
            the largest finite value representable by `dtype`.
        neginf: The value used to replace negative infinity. Defaults to
            the most negative finite value representable by `dtype`.

    Returns:
        An array with the same shape as `a`, with `NaN` and infinite
        elements replaced by finite values.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.array[f64]("[1.0, nan, inf, -inf]")
        print(nm.nan_to_num(a))  # [1.0, 0.0, 1.7976931348623157e+308, -1.7976931348623157e+308]
        ```
    """

    var pos_value = posinf.value() if posinf else max_finite[dtype]()
    var neg_value = neginf.value() if neginf else min_finite[dtype]()

    var result = NDArray[dtype](a.shape)
    for i in range(a.size):
        var value = a.item(i)
        if math.isnan(value):
            result.itemset(i, nan)
        elif math.isinf(value):
            result.itemset(i, pos_value if value > 0 else neg_value)
        else:
            result.itemset(i, value)

    return result^
