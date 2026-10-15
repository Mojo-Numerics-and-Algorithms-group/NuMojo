# ===----------------------------------------------------------------------=== #
# NuMojo: Convolution and interpolation
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Signal routines (numojo.routines.math.signal).
==============================================
Discrete convolution and one-dimensional linear interpolation.

Exports
-------
- `convolve`: Discrete, linear convolution of two 1-d arrays.
- `interp`: One-dimensional linear interpolation.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.manipulation import ravel


def convolve[
    dtype: DType, //, returned_dtype: DType = DType.float64
](
    a: NDArray[dtype], v: NDArray[dtype], mode: String = "full"
) raises -> NDArray[returned_dtype]:
    """
    Compute the discrete, linear convolution of two 1-d arrays.

    Matches `numpy.convolve`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A 1-d array.
        v: A 1-d array.
        mode: One of `"full"` (the default), `"same"`, or `"valid"`.

    Returns:
        The convolution of `a` and `v`. For `"full"`, the length is
        `a.size + v.size - 1`; for `"same"`, `max(a.size, v.size)`;
        for `"valid"`, `max(a.size, v.size) - min(a.size, v.size) + 1`.

    Raises:
        NumojoError: If `a` or `v` is not 1-dimensional, or either is
            empty.
        NumojoError: If `mode` is not one of `"full"`, `"same"`, or
            `"valid"`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, 2.0, 3.0]")
        var v = nm.fromstring("[0.0, 1.0, 0.5]")
        print(nm.convolve(a, v))  # [0.0, 1.0, 2.5, 4.0, 1.5]
    ```
    """

    if (a.ndim != 1) or (v.ndim != 1):
        raise Error(
            NumojoError(
                category="shape",
                message="`convolve` only supports 1-dimensional arrays.",
                location="convolve",
            )
        )
    if (a.size == 0) or (v.size == 0):
        raise Error(
            NumojoError(
                category="value",
                message="`convolve` does not support empty arrays.",
                location="convolve",
            )
        )
    if (mode != "full") and (mode != "same") and (mode != "valid"):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    '`mode` must be "full", "same", or "valid", got "{}"'
                ).format(mode),
                location="convolve",
            )
        )

    var m = a.size
    var n = v.size
    var full_len = m + n - 1
    var full = NDArray[returned_dtype](Shape(full_len))

    for k in range(full_len):
        var total = Scalar[returned_dtype](0)
        var lo = max(0, k - n + 1)
        var hi = min(k, m - 1)
        for i in range(lo, hi + 1):
            total += (
                a.item(i).cast[returned_dtype]()
                * v.item(k - i).cast[returned_dtype]()
            )
        full[Item(k)] = total

    if mode == "full":
        return full^

    var short_len = min(m, n)
    var long_len = max(m, n)
    var out_len: Int
    var offset: Int
    if mode == "same":
        out_len = long_len
        offset = (full_len - out_len) // 2
    else:  # valid
        out_len = long_len - short_len + 1
        offset = short_len - 1

    var result = NDArray[returned_dtype](Shape(out_len))
    for i in range(out_len):
        result[Item(i)] = full.item(offset + i)

    return result^


def interp[
    dtype: DType,
    xpdtype: DType,
    fpdtype: DType,
    //,
    returned_dtype: DType = DType.float64,
](
    x: Scalar[dtype],
    xp: NDArray[xpdtype],
    fp: NDArray[fpdtype],
    left: Optional[Scalar[returned_dtype]] = None,
    right: Optional[Scalar[returned_dtype]] = None,
) raises -> Scalar[returned_dtype]:
    """
    One-dimensional linear interpolation of a scalar sample point.

    Matches `numpy.interp` for a scalar `x`. `xp` must be sorted in
    increasing order.

    Parameters:
        dtype: The element type of `x`.
        xpdtype: The element type of `xp`.
        fpdtype: The element type of `fp`.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        x: The sample point at which to interpolate.
        xp: The x-coordinates of the data points, increasing.
        fp: The y-coordinates of the data points, the same length as
            `xp`.
        left: The value to return for `x < xp[0]`. Defaults to
            `fp[0]`.
        right: The value to return for `x > xp[-1]`. Defaults to
            `fp[-1]`.

    Returns:
        The interpolated value at `x`.

    Raises:
        NumojoError: If `xp` and `fp` are not 1-dimensional and the
            same length, or if `xp` is empty.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var xp = nm.fromstring("[1.0, 2.0, 3.0]")
        var fp = nm.fromstring("[3.0, 2.0, 0.0]")
        print(nm.interp(Float64(2.5), xp, fp))  # 1.0
    ```
    """

    if (xp.ndim != 1) or (fp.ndim != 1) or (xp.size != fp.size):
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "`xp` and `fp` must be 1-dimensional and the same length."
                ),
                location="interp",
            )
        )
    if xp.size == 0:
        raise Error(
            NumojoError(
                category="value",
                message="`xp` must not be empty.",
                location="interp",
            )
        )

    var n = xp.size
    var xf = x.cast[returned_dtype]()
    var xp0 = xp.item(0).cast[returned_dtype]()
    var xpn = xp.item(n - 1).cast[returned_dtype]()

    if xf < xp0:
        return left.value() if left else fp.item(0).cast[returned_dtype]()
    if xf > xpn:
        return right.value() if right else fp.item(n - 1).cast[returned_dtype]()

    var lo = 0
    var hi = n - 1
    while lo < hi - 1:
        var mid = (lo + hi) // 2
        if xp.item(mid).cast[returned_dtype]() <= xf:
            lo = mid
        else:
            hi = mid

    var x_lo = xp.item(lo).cast[returned_dtype]()
    var x_hi = xp.item(hi).cast[returned_dtype]()
    var f_lo = fp.item(lo).cast[returned_dtype]()
    var f_hi = fp.item(hi).cast[returned_dtype]()

    if x_hi == x_lo:
        return f_lo
    return f_lo + (f_hi - f_lo) * (xf - x_lo) / (x_hi - x_lo)


def interp[
    dtype: DType,
    xpdtype: DType,
    fpdtype: DType,
    //,
    returned_dtype: DType = DType.float64,
](
    x: NDArray[dtype],
    xp: NDArray[xpdtype],
    fp: NDArray[fpdtype],
    left: Optional[Scalar[returned_dtype]] = None,
    right: Optional[Scalar[returned_dtype]] = None,
) raises -> NDArray[returned_dtype]:
    """
    One-dimensional linear interpolation at an array of sample points.

    Matches `numpy.interp`. `xp` must be sorted in increasing order.

    Parameters:
        dtype: The element type of `x`.
        xpdtype: The element type of `xp`.
        fpdtype: The element type of `fp`.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        x: An array of sample points. It is treated as flat
            regardless of its shape.
        xp: The x-coordinates of the data points, increasing.
        fp: The y-coordinates of the data points, the same length as
            `xp`.
        left: The value to return for `x < xp[0]`. Defaults to
            `fp[0]`.
        right: The value to return for `x > xp[-1]`. Defaults to
            `fp[-1]`.

    Returns:
        The interpolated values, flat and the same length as `x`.

    Raises:
        NumojoError: If `xp` and `fp` are not 1-dimensional and the
            same length, or if `xp` is empty.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var x = nm.fromstring("[0.0, 1.5, 4.0]")
        var xp = nm.fromstring("[1.0, 2.0, 3.0]")
        var fp = nm.fromstring("[3.0, 2.0, 0.0]")
        print(nm.interp(x, xp, fp))  # [3.0, 1.5, 0.0]
    ```
    """

    var flat_x = ravel(x)
    var result = NDArray[returned_dtype](Shape(flat_x.size))
    for i in range(flat_x.size):
        result[Item(i)] = interp[returned_dtype=returned_dtype](
            flat_x.item(i), xp, fp, left, right
        )
    return result^
