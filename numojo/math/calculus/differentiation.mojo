# ===----------------------------------------------------------------------=== #
# implements basic Integral functions
# Last updated: 2024-07-02
# ===----------------------------------------------------------------------=== #

import math
import .. math_funcs as _mf
import ... core as core
from ...core.ndarray import NDArray, NDArrayShape
from ...core.utility_funcs import is_inttype, is_floattype

"""TODO: 
1) add a Variant[NDArray, Scalar, ...] to include all possibilities
2) add edge_order
"""


fn gradient[
    dtype: DType
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
    
    var result: NDArray[dtype] = NDArray[dtype](x.shape(), random=False)
    var space: NDArray[dtype] = core.arange[dtype](
        1, x.num_elements() + 1, step=spacing
    )
    var hu: Scalar[dtype] = space.get_scalar(1)
    var hd: Scalar[dtype] = space.get_scalar(0)
    result.store(
        0,
        (x.get_scalar(1) - x.get_scalar(0))
        / (hu - hd),
    )

    hu = space.get_scalar(x.num_elements() - 1)
    hd = space.get_scalar(x.num_elements() - 2)
    result.store(
        x.num_elements() - 1,
        (
            x.get_scalar(x.num_elements() - 1)
            - x.get_scalar(x.num_elements() - 2)
        )
        / (hu - hd),
    )

    for i in range(1, x.num_elements() - 1):
        var hu: Scalar[dtype] = space.get_scalar(i + 1) - space.get_scalar(
            i
        )
        var hd: Scalar[dtype] = space.get_scalar(i) - space.get_scalar(
            i - 1
        )
        var fi: Scalar[dtype] = (
            hd**2 * x.get_scalar(i + 1)
            + (hu**2 - hd**2) * x.get_scalar(i)
            - hu**2 * x.get_scalar(i - 1)
        ) / (hu * hd * (hu + hd))
        result.store(i, fi)

    return result
