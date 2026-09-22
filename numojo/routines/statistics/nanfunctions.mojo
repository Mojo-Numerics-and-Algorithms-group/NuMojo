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
from numojo.routines.functional import apply_along_axis_reduce
from numojo.routines.manipulation import ravel


def nansum_1d[
    dtype: DType, //
](a: NDArray[dtype]) capturing raises -> Scalar[
    dtype
] where dtype.is_floating_point():
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
](a: NDArray[dtype]) capturing raises -> Scalar[
    dtype
] where dtype.is_floating_point():
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
](a: NDArray[dtype]) capturing raises -> Scalar[
    dtype
] where dtype.is_floating_point():
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
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Find the max value in a 1-D array, ignoring `NaN` elements."""
    return nanextrema_1d[is_max=True](a)


def nanextrema_1d_min[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[
    dtype
] where dtype.is_floating_point():
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
