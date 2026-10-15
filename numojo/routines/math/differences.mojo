# ===----------------------------------------------------------------------=== #
# NuMojo: Difference routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Differences (numojo.routines.math.differences).
===============================================
Numerical differentiation and integration helpers.

Implements gradient computation and finite differences for numerical differentiation
and integration tasks.

Exports
-------
- `gradient`: Compute gradients using the trapezoidal rule.
- `diff`: Compute n-th order finite differences.
- `ediff1d`: Compute the differences of a flattened array, with
  optional values prepended/appended.
"""

# TODO: Add Variant[NDArray, Scalar, ...] to include all possibilities
# TODO: Add edge_order parameter support

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math
from max.algorithm import parallelize

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.indexing.item import Item
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import arange
from numojo.routines.manipulation import ravel

# ===----------------------------------------------------------------------=== #
# Gradient computation using the trapezoidal rule.
# ===----------------------------------------------------------------------=== #


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


# ===----------------------------------------------------------------------=== #
# Differences
# ===----------------------------------------------------------------------=== #


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

    var current: NDArray[dtype] = array.copy()

    for _ in range(n):
        var result: NDArray[dtype] = NDArray[dtype](
            NDArrayShape(current.size - 1)
        )
        for i in range(current.size - 1):
            result.store(i, current.load(i + 1) - current.load(i))
        current = result^
    return current^


# ===----------------------------------------------------------------------=== #
# Flattened differences
# ===----------------------------------------------------------------------=== #


def ediff1d[
    dtype: DType
](
    ary: NDArray[dtype],
    to_end: Optional[NDArray[dtype]] = None,
    to_begin: Optional[NDArray[dtype]] = None,
) raises -> NDArray[dtype]:
    """
    Compute the consecutive differences of a flattened array, with
    optional values prepended and/or appended.

    Matches `numpy.ediff1d`.

    Parameters:
        dtype: The element type.

    Args:
        ary: An array. It is treated as flat regardless of its shape.
        to_end: Values to append to the end of the differences.
        to_begin: Values to prepend to the start of the differences.

    Returns:
        An array of consecutive differences, of length
        `ary.size - 1 + to_begin.size + to_end.size`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, 2.0, 4.0, 7.0]")
        print(nm.ediff1d(a))  # [1.0, 2.0, 3.0]
    ```
    """

    var flat = ravel(ary)
    var n_diff = max(flat.size - 1, 0)
    var n_begin = to_begin.value().size if to_begin else 0
    var n_end = to_end.value().size if to_end else 0

    var result = NDArray[dtype](Shape(n_begin + n_diff + n_end))

    var offset = 0
    if to_begin:
        var tb = to_begin.value().copy()
        for i in range(tb.size):
            result[Item(offset + i)] = tb.item(i)
        offset += tb.size

    for i in range(n_diff):
        result[Item(offset + i)] = flat.item(i + 1) - flat.item(i)
    offset += n_diff

    if to_end:
        var te = to_end.value().copy()
        for i in range(te.size):
            result[Item(offset + i)] = te.item(i)

    return result^
