# ===----------------------------------------------------------------------=== #
# NuMojo: Difference routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Difference routines for NuMojo (numojo.routines.math.differences).

Implements gradient and trapezoidal integration helpers for numerical differentiation and integration tasks.
"""

import std.math
from std.algorithm import parallelize

from numojo.routines.creation import arange
from numojo.core.ndarray import NDArray
from numojo.core.dtype.utility import is_inttype, is_floattype

# TODO:
# 1) add a Variant[NDArray, Scalar, ...] to include all possibilities
# 2) add edge_order

# ===------------------------------------------------------------------------===#
# Gradient computation using the trapezoidal rule.
# ===------------------------------------------------------------------------===#


def gradient[
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

    # View safety guard: ensure input is C-contiguous before linear access.
    if not x.is_c_contiguous():
        return gradient[dtype](x.contiguous(), spacing)

    var space: NDArray[dtype] = arange[dtype](
        1, Scalar[dtype](x.size + 1), step=spacing
    )
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


# ===------------------------------------------------------------------------===#
# Differences
# ===------------------------------------------------------------------------===#


def diff[
    dtype: DType = DType.float64
](array: NDArray[dtype], n: Int = 1) raises -> NDArray[dtype]:
    """
    Compute the n-th order difference of the input array.

    Parameters:
        dtype: The element type.

    Args:
        array: A array.
        n: The order of the difference.

    Returns:
        The n-th order difference of the input array.
    """

    var current: NDArray[dtype] = array.deep_copy()

    for _ in range(n):
        var result: NDArray[dtype] = NDArray[dtype](
            NDArrayShape(current.size - 1)
        )
        for i in range(current.size - 1):
            result.store(i, current.load(i + 1) - current.load(i))
        current = result^
    return current^
