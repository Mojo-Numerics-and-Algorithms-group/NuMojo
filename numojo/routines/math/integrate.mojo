# ===----------------------------------------------------------------------=== #
# NuMojo: Numerical integration
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Integration (numojo.routines.math.integrate).
=============================================
Trapezoidal-rule integration of flattened arrays.

Exports
-------
- `trapezoid`: Definite integral using the composite trapezoidal rule.
- `cumtrapz`: Cumulative integral using the composite trapezoidal rule.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.manipulation import ravel


def trapezoid[
    dtype: DType, //, returned_dtype: DType = DType.float64
](y: NDArray[dtype], dx: Scalar[returned_dtype] = 1.0) raises -> Scalar[
    returned_dtype
]:
    """
    Integrate a flattened array using the composite trapezoidal rule,
    with uniformly-spaced samples.

    Matches `numpy.trapezoid` with the default `dx` and
    no `x` array.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        y: An array of sample values. It is treated as flat regardless
            of its shape.
        dx: The uniform spacing between samples.

    Returns:
        The definite integral approximated by the trapezoidal rule.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var y = nm.fromstring("[1.0, 2.0, 3.0]")
        print(nm.trapezoid(y))  # 4.0
    ```
    """

    var flat = ravel(y)
    var total = Scalar[returned_dtype](0)
    for i in range(flat.size - 1):
        total += (
            flat.item(i).cast[returned_dtype]()
            + flat.item(i + 1).cast[returned_dtype]()
        )
    return total * dx / 2


def trapezoid[
    dtype: DType, xdtype: DType, //, returned_dtype: DType = DType.float64
](y: NDArray[dtype], x: NDArray[xdtype]) raises -> Scalar[returned_dtype]:
    """
    Integrate a flattened array using the composite trapezoidal rule,
    with non-uniformly-spaced sample positions.

    Matches `numpy.trapezoid` with an explicit `x`.

    Parameters:
        dtype: The element type of `y`.
        xdtype: The element type of `x`.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        y: An array of sample values. It is treated as flat regardless
            of its shape.
        x: The sample positions, the same length as `y`.

    Returns:
        The definite integral approximated by the trapezoidal rule.

    Raises:
        NumojoError: If `x` is not the same length as `y`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var y = nm.fromstring("[1.0, 2.0, 3.0]")
        var x = nm.fromstring("[0.0, 1.0, 2.0]")
        print(nm.trapezoid(y, x))  # 4.0
    ```
    """

    var flat_y = ravel(y)
    var flat_x = ravel(x)
    if flat_x.size != flat_y.size:
        raise Error(
            NumojoError(
                category="shape",
                message="`x` must be the same length as `y`.",
                location="trapezoid",
            )
        )

    var total = Scalar[returned_dtype](0)
    for i in range(flat_y.size - 1):
        var dx = (
            flat_x.item(i + 1).cast[returned_dtype]()
            - flat_x.item(i).cast[returned_dtype]()
        )
        total += (
            dx
            * (
                flat_y.item(i).cast[returned_dtype]()
                + flat_y.item(i + 1).cast[returned_dtype]()
            )
            / 2
        )
    return total


def cumtrapz[
    dtype: DType, //, returned_dtype: DType = DType.float64
](
    y: NDArray[dtype],
    dx: Scalar[returned_dtype] = 1.0,
    initial: Optional[Scalar[returned_dtype]] = None,
) raises -> NDArray[returned_dtype]:
    """
    Cumulatively integrate a flattened array using the composite
    trapezoidal rule, with uniformly-spaced samples.

    Matches `scipy.integrate.cumulative_trapezoid` with the default
    `dx` and no `x` array.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        y: An array of sample values. It is treated as flat regardless
            of its shape.
        dx: The uniform spacing between samples.
        initial: If given, prepended to the result so its length
            matches `y`. If `None` (the default), the result has
            length `y.size - 1`.

    Returns:
        The cumulative integral approximated by the trapezoidal rule.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var y = nm.fromstring("[1.0, 2.0, 3.0]")
        print(nm.cumtrapz(y))  # [1.5, 4.0]
    ```
    """

    var flat = ravel(y)
    var n_steps = max(flat.size - 1, 0)
    var offset = 1 if initial else 0
    var result = NDArray[returned_dtype](Shape(n_steps + offset))

    if initial:
        result[Item(0)] = initial.value()

    var running = Scalar[returned_dtype](0)
    for i in range(n_steps):
        running += (
            dx
            * (
                flat.item(i).cast[returned_dtype]()
                + flat.item(i + 1).cast[returned_dtype]()
            )
            / 2
        )
        result[Item(offset + i)] = running

    return result^


def cumtrapz[
    dtype: DType, xdtype: DType, //, returned_dtype: DType = DType.float64
](
    y: NDArray[dtype],
    x: NDArray[xdtype],
    initial: Optional[Scalar[returned_dtype]] = None,
) raises -> NDArray[returned_dtype]:
    """
    Cumulatively integrate a flattened array using the composite
    trapezoidal rule, with non-uniformly-spaced sample positions.

    Matches `scipy.integrate.cumulative_trapezoid` with an explicit
    `x`.

    Parameters:
        dtype: The element type of `y`.
        xdtype: The element type of `x`.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        y: An array of sample values. It is treated as flat regardless
            of its shape.
        x: The sample positions, the same length as `y`.
        initial: If given, prepended to the result so its length
            matches `y`. If `None` (the default), the result has
            length `y.size - 1`.

    Returns:
        The cumulative integral approximated by the trapezoidal rule.

    Raises:
        NumojoError: If `x` is not the same length as `y`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var y = nm.fromstring("[1.0, 2.0, 3.0]")
        var x = nm.fromstring("[0.0, 1.0, 2.0]")
        print(nm.cumtrapz(y, x))  # [1.5, 4.0]
    ```
    """

    var flat_y = ravel(y)
    var flat_x = ravel(x)
    if flat_x.size != flat_y.size:
        raise Error(
            NumojoError(
                category="shape",
                message="`x` must be the same length as `y`.",
                location="cumtrapz",
            )
        )

    var n_steps = max(flat_y.size - 1, 0)
    var offset = 1 if initial else 0
    var result = NDArray[returned_dtype](Shape(n_steps + offset))

    if initial:
        result[Item(0)] = initial.value()

    var running = Scalar[returned_dtype](0)
    for i in range(n_steps):
        var dx = (
            flat_x.item(i + 1).cast[returned_dtype]()
            - flat_x.item(i).cast[returned_dtype]()
        )
        running += (
            dx
            * (
                flat_y.item(i).cast[returned_dtype]()
                + flat_y.item(i + 1).cast[returned_dtype]()
            )
            / 2
        )
        result[Item(offset + i)] = running

    return result^
