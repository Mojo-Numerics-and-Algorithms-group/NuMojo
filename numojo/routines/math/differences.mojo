# ===----------------------------------------------------------------------=== #
# Differences
# ===----------------------------------------------------------------------=== #

import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc

import numojo.core._math_funcs as _mf
from numojo.routines.creation import arange
from numojo.core.ndarray import NDArray
from numojo.core.utility import is_inttype, is_floattype

# TODO:
# 1) add a Variant[NDArray, Scalar, ...] to include all possibilities
# 2) add edge_order


fn gradient[
    dtype: DType = DType.float64
](x: NDArray[dtype], spacing: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Compute the gradient of y over x using the trapezoidal rule.

    Parameters:
        dtype: Input data type.

    Args:
        x: An array.
        spacing: An array of the same shape as x containing the spacing between adjacent elements.

    Constraints:
        `fdtype` must be a floating-point type if `idtype` is not a floating-point type.

    Returns:
        The integral of y over x using the trapezoidal rule.
    """

    var result: NDArray[dtype] = NDArray[dtype](x.shape)
    var space: NDArray[dtype] = arange[dtype](1, x.size + 1, step=spacing)
    var hu: Scalar[dtype] = space.load(1)
    var hd: Scalar[dtype] = space.load(0)
    result.store(
        0,
        (x.load(1) - x.load(0)) / (hu - hd),
    )

    hu = space.load(x.size - 1)
    hd = space.load(x.size - 2)
    result.store(
        x.size - 1,
        (x.load(x.size - 1) - x.load(x.size - 2)) / (hu - hd),
    )

    for i in range(1, x.size - 1):
        var hu: Scalar[dtype] = space.load(i + 1) - space.load(i)
        var hd: Scalar[dtype] = space.load(i) - space.load(i - 1)
        var fi: Scalar[dtype] = (
            hd**2 * x.load(i + 1)
            + (hu**2 - hd**2) * x.load(i)
            - hu**2 * x.load(i - 1)
        ) / (hu * hd * (hu + hd))
        result.store(i, fi)

    return result^


# naive loop implementation, optimize later
fn trapz[
    dtype: DType = DType.float64
](y: NDArray[dtype], x: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Compute the integral of y over x using the trapezoidal rule.

    Parameters:
        dtype: The element type.

    Args:
        y: An array.
        x: An array.

    Constraints:
        `x` and `y` must have the same shape.
        `fdtype` must be a floating-point type if `idtype` is not a floating-point type.

    Returns:
        The integral of y over x using the trapezoidal rule.
    """
    constrained[
        is_inttype[dtype]() and not is_floattype[dtype](),
        (
            "output dtype `Fdtype` must be a floating-point type if input dtype"
            " `Idtype` is not a floating-point type"
        ),
    ]()

    if x.shape != y.shape:
        raise Error("x and y must have the same shape")

    var integral: Scalar[dtype] = 0.0
    for i in range(x.size - 1):
        var temp = (x.load(i + 1) - x.load(i)) * (
            y.load(i) + y.load(i + 1)
        ) / 2.0
        integral += temp
    return integral
