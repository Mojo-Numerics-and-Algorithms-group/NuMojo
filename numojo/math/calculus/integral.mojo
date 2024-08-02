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
    in_dtype: DType, out_dtype: DType = DType.float32
](y: NDArray[in_dtype], x: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Compute the integral of y over x using the trapezoidal rule.

    Parameters:
        in_dtype: Input data type.
        out_dtype: Output data type, defaults to float32.

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
    if is_inttype[in_dtype]() and not is_floattype[out_dtype]():
        raise Error(
            "output dtype `Fdtype` must be a floating-point type if input dtype"
            " `Idtype` is not a floating-point type"
        )

    var integral: SIMD[out_dtype] = 0.0
    for i in range(x.num_elements() - 1):
        var temp = (x.get_scalar(i + 1) - x.get_scalar(i)).cast[out_dtype]() * (
            y.get_scalar(i) + y.get_scalar(i + 1)
        ).cast[out_dtype]() / 2.0
        integral += temp
    return integral
