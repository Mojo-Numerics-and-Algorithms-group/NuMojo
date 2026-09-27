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
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import _0darray
from numojo.routines.functional import apply_along_axis_reduce
from numojo.routines.manipulation import ravel
from numojo.routines.sorting import sort


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
