"""
# ===----------------------------------------------------------------------=== #
# Interpolate Module - Implements interpolation functions
# Last updated: 2024-06-14
# ===----------------------------------------------------------------------=== #
"""

from .core.ndarray import NDArray, NDArrayShape

"""
# TODO:
1) Cross check all the functions with numpy
2) Add support for axis argument
"""


fn interp1d[
    dtype: DType = DType.float64
](
    xi: NDArray[dtype],
    x: NDArray[dtype],
    y: NDArray[dtype],
    method: String = "linear",
    fill_value: String = "interpolate",
) raises -> NDArray[dtype]:
    """
    Interpolate the values of y at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
        method: The interpolation method.
        fill_value: The fill value.
    Returns:
        The interpolated values of y at the points xi as An Array of `dtype`.
    """
    # linear
    if method == "linear" and fill_value == "extrapolate":
        return _interp1d_linear_extrapolate(xi, x, y)
    elif method == "linear" and fill_value == "interpolate":
        return _interp1d_linear_interpolate(xi, x, y)

    # quadratic
    elif method == "quadratic" and fill_value == "extrapolate":
        return _interp1d_quadratic_extrapolate(xi, x, y)
    elif method == "quadratic" and fill_value == "interpolate":
        return _interp1d_quadratic_interpolate(xi, x, y)

    # cubic
    elif method == "cubic" and fill_value == "extrapolate":
        return _interp1d_cubic_extrapolate(xi, x, y)
    elif method == "cubic" and fill_value == "interpolate":
        return _interp1d_cubic_interpolate(xi, x, y)

    else:
        print("Invalid interpolation method: " + method)
        return NDArray[dtype]()


fn _interp1d_linear_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Linear interpolation of the (x, y) values at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
    Returns:
        The linearly interpolated values of y at the points xi as An Array of `dtype`.
    """
    var result = NDArray[dtype](xi.shape())
    for i in range(xi.num_elements()):
        if xi[i] <= x[0]:
            result[i] = y[0]
        elif xi[i] >= x[x.num_elements() - 1]:
            result[i] = y[y.num_elements() - 1]
        else:
            var j = 0
            while xi[i] > x[j]:
                j += 1
            var x0 = x[j - 1]
            var x1 = x[j]
            var y0 = y[j - 1]
            var y1 = y[j]
            var t = (xi[i] - x0) / (x1 - x0)
            result[i] = y0 + t * (y1 - y0)
    return result


fn _interp1d_linear_extrapolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Linear extrapolation of the (x, y) values at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
    Returns:
        The linearly extrapolated values of y at the points xi as An Array of `dtype`.
    """
    var result = NDArray[dtype](xi.shape())
    for i in range(xi.num_elements()):
        if xi[i] <= x[0]:
            var slope = (y[1] - y[0]) / (x[1] - x[0])
            result[i] = y[0] + slope * (xi[i] - x[0])
        elif xi[i] >= x[x.num_elements() - 1]:
            var slope = (y[y.num_elements() - 1] - y[y.num_elements() - 2]) / (
                x[x.num_elements() - 1] - x[x.num_elements() - 2]
            )
            result[i] = y[y.num_elements() - 1] + slope * (
                xi[i] - x[x.num_elements() - 1]
            )
        else:
            var j = 0
            while xi[i] > x[j]:
                j += 1
            var x0 = x[j - 1]
            var x1 = x[j]
            var y0 = y[j - 1]
            var y1 = y[j]
            var t = (xi[i] - x0) / (x1 - x0)
            result[i] = y0 + t * (y1 - y0)
    return result


fn _interp1d_quadratic_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Quadratic interpolation of the (x, y) values at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
    Returns:
        The quadratically interpolated values of y at the points xi as An Array of `dtype`.
    """
    var result = NDArray[dtype](xi.shape())
    for i in range(xi.num_elements()):
        if xi[i] <= x[0]:
            result[i] = y[0]
        elif xi[i] >= x[x.num_elements() - 1]:
            result[i] = y[y.num_elements() - 1]
        else:
            var j = 1
            while xi[i] > x[j]:
                j += 1
            var x0 = x[j - 2]
            var x1 = x[j - 1]
            var x2 = x[j]
            var y0 = y[j - 2]
            var y1 = y[j - 1]
            var y2 = y[j]
            var t = (xi[i] - x1) / (x2 - x1)
            var a = y0
            var b = y1
            var c = y2
            result[i] = a * t * t + b * t + c
    return result


fn _interp1d_quadratic_extrapolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Quadratic extrapolation of the (x, y) values at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
    Returns:
        The quadratically extrapolated values of y at the points xi as An Array of `dtype`.
    """
    var result = NDArray[dtype](xi.shape())
    for i in range(xi.num_elements()):
        if xi[i] <= x[0]:
            var slope = (y[1] - y[0]) / (x[1] - x[0])
            var intercept = y[0] - slope * x[0]
            result[i] = intercept + slope * xi[i]
        elif xi[i] >= x[x.num_elements() - 1]:
            var slope = (y[y.num_elements() - 1] - y[y.num_elements() - 2]) / (
                x[x.num_elements() - 1] - x[x.num_elements() - 2]
            )
            var intercept = y[y.num_elements() - 1] - slope * x[
                x.num_elements() - 1
            ]
            result[i] = intercept + slope * xi[i]
        else:
            var j = 1
            while xi[i] > x[j]:
                j += 1
            var x0 = x[j - 2]
            var x1 = x[j - 1]
            var x2 = x[j]
            var y0 = y[j - 2]
            var y1 = y[j - 1]
            var y2 = y[j]
            var t = (xi[i] - x1) / (x2 - x1)
            var a = y0
            var b = y1
            var c = y2
            result[i] = a * t * t + b * t + c
    return result


fn _interp1d_cubic_interpolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Cubic interpolation of the (x, y) values at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
    Returns:
        The cubically interpolated values of y at the points xi as An Array of `dtype`.
    """
    var result = NDArray[dtype](xi.shape())
    for i in range(xi.num_elements()):
        if xi[i] <= x[0]:
            result[i] = y[0]
        elif xi[i] >= x[x.num_elements() - 1]:
            result[i] = y[y.num_elements() - 1]
        else:
            var j = 0
            while xi[i] > x[j]:
                j += 1
            # Ensure we have enough points for cubic interpolation
            # var j = math.max(j, 2)
            # var j = math.min(j, x.num_elements() - 2)

            var x0 = x[j - 2]
            var x1 = x[j - 1]
            var x2 = x[j]
            var x3 = x[j + 1]

            var y0 = y[j - 2]
            var y1 = y[j - 1]
            var y2 = y[j]
            var y3 = y[j + 1]

            var t = (xi[i] - x1) / (x2 - x1)

            # Cubic interpolation formula
            var a0 = y3 - y2 - y0 + y1
            var a1 = y0 - y1 - a0
            var a2 = y2 - y0
            var a3 = y1

            result[i] = a0 * t * t * t + a1 * t * t + a2 * t + a3
    return result


fn _interp1d_cubic_extrapolate[
    dtype: DType
](xi: NDArray[dtype], x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Cubic extrapolation of the (x, y) values at the points xi.
    Parameters:
        dtype: The element type.
    Args:
        xi: An Array.
        x: An Array.
        y: An Array.
    Returns:
        The cubically extrapolated values of y at the points xi as An Array of `dtype`.
    """
    var result = NDArray[dtype](xi.shape())
    for i in range(xi.num_elements()):
        if xi[i] <= x[0]:
            var t = (xi[i] - x[0]) / (x[1] - x[0])
            var a0 = y[2] - y[1] - y[0] + y[1]
            var a1 = y[0] - y[1] - a0
            var a2 = y[1] - y[0]
            var a3 = y[0]
            result[i] = a0 * t * t * t + a1 * t * t + a2 * t + a3
        elif xi[i] >= x[x.num_elements() - 1]:
            var t = (xi[i] - x[x.num_elements() - 2]) / (
                x[x.num_elements() - 1] - x[x.num_elements() - 2]
            )
            var a0 = y[y.num_elements() - 1] - y[y.num_elements() - 2] - y[
                y.num_elements() - 3
            ] + y[y.num_elements() - 2]
            var a1 = y[y.num_elements() - 3] - y[y.num_elements() - 2] - a0
            var a2 = y[y.num_elements() - 2] - y[y.num_elements() - 3]
            var a3 = y[y.num_elements() - 2]
            result[i] = a0 * t * t * t + a1 * t * t + a2 * t + a3
        else:
            var j = 1
            while xi[i] > x[j]:
                j += 1
            var x0 = x[j - 2]
            var x1 = x[j - 1]
            var x2 = x[j]
            var x3 = x[j + 1]
            var y0 = y[j - 2]
            var y1 = y[j - 1]
            var y2 = y[j]
            var y3 = y[j + 1]
            var t = (xi[i] - x1) / (x2 - x1)
            var a0 = y3 - y2 - y0 + y1
            var a1 = y0 - y1 - a0
            var a2 = y2 - y0
            var a3 = y1
            result[i] = a0 * t * t * t + a1 * t * t + a2 * t + a3
    return result
