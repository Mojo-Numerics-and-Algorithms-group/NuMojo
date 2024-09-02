# ===----------------------------------------------------------------------=== #
# implements basic Integral functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #

import math
import .. math_funcs as _mf
from ...core.ndarray import NDArray, NDArrayShape
from ...core.utility_funcs import is_inttype, is_floattype
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc


# naive loop implementation, optimize later
fn trapz[
    dtype: DType = DType.float64
](y: NDArray[dtype], x: NDArray[dtype]) raises -> SIMD[dtype, 1]:
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
    if x.shape() != y.shape():
        raise Error("x and y must have the same shape")

    # move this check to compile time using constrained?
    if is_inttype[dtype]() and not is_floattype[dtype]():
        raise Error(
            "output dtype `Fdtype` must be a floating-point type if input dtype"
            " `Idtype` is not a floating-point type"
        )

    var integral: SIMD[dtype, 1] = 0.0
    for i in range(x.num_elements() - 1):
        var temp = (x.get_scalar(i + 1) - x.get_scalar(i)) * (
            y.get_scalar(i) + y.get_scalar(i + 1)
        ) / 2.0
        integral += temp
    return integral
