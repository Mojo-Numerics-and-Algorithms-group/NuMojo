# ===----------------------------------------------------------------------=== #
# NuMojo: Logical ops
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Logical Operations (numojo.routines.logic.logical_ops).
=======================================================
Element-wise logical operations for arrays.

Implements logical AND, OR, XOR, and NOT operations for NDArray and
ComplexNDArray types.

Exports
-------
- `logical_and`: Logical AND operation.
- `logical_or`: Logical OR operation.
- `logical_xor`: Logical XOR operation.
- `logical_not`: Logical NOT operation.
"""

# TODO: Add `where` argument support to logical operations.
# TODO: Create backend for these operations.
# FIXME: Make all SIMD vectorized once bool bit-packing issue is resolved.

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.complex.complex_ndarray import ComplexNDArray
from numojo.core.dtype.complex_dtype import ComplexDType
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===----------------------------------------------------------------------=== #
# Logical operations for NDArray
# ===----------------------------------------------------------------------=== #
def logical_and[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical AND operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Raises:
        - NumojoError: If the input arrays do not have the same shape.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        An array containing the result of the logical AND operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_and

        var a = nm.arange(0, 10)
        var b = nm.arange(5, 15)
        var result = logical_and(a > 3, b < 10)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input arrays must have the same shape for logical AND"
                    " operation."
                ),
                location="numojo.routines.logic.logical_and",
            )
        )

    @parameter
    def kernel[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
        return SIMD[DType.bool, width](a & b)

    return HostExecutor.apply_binary_predicate[dtype, kernel](a, b)


def logical_or[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical OR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Raises:
        - NumojoError: If the input arrays do not have the same shape.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        An array containing the result of the logical OR operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_or

        var a = nm.arange(0, 10)
        var b = nm.arange(5, 15)
        var result = logical_or(a < 3, b > 10)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input arrays must have the same shape for logical OR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_or",
            )
        )

    @parameter
    def kernel[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
        return SIMD[DType.bool, width](a | b)

    return HostExecutor.apply_binary_predicate[dtype, kernel](a, b)


def logical_not[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical NOT operation on an array.

    Args:
        a: Input array.

    Raises:
        - NumojoError: If the input array is not of a supported data type.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        An array containing the result of the logical NOT operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_not

        var a = nm.arange(0, 10)
        var result = logical_not(a < 5)
        ```
    """

    @parameter
    def kernel[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
        return SIMD[DType.bool, width](~a)

    return HostExecutor.apply_unary_predicate[dtype, kernel](a)


def logical_xor[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical XOR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Raises:
        - NumojoError: If the input arrays do not have the same shape.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        An array containing the result of the logical XOR operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_xor

        var a = nm.arange(0, 10)
        var b = nm.arange(5, 15)
        var result = logical_xor(a > 3, b < 10)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input arrays must have the same shape for logical XOR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_xor",
            )
        )

    @parameter
    def kernel[
        dtype: DType, width: Int
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
        return SIMD[DType.bool, width](a ^ b)

    return HostExecutor.apply_binary_predicate[dtype, kernel](a, b)


# ===----------------------------------------------------------------------=== #
# Logical operations for ComplexNDArray
# ===----------------------------------------------------------------------=== #


def logical_and[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype.dtype == DType.bool or cdtype.dtype.is_integral()
):
    """
    Element-wise logical AND operation between two complex arrays.

    Args:
        a: First input complex array.
        b: Second input complex array.

    Raises:
        - NumojoError: If the input arrays do not have the same shape.

    Constraints:
        - Supports only boolean and integral complex data types.

    Returns:
        A complex array containing the result of the logical AND operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_and

        var a = nm.arange[ci32](CScalar[ci32](0), CScalar[ci32](10))
        var b = nm.arange[ci32](CScalar[ci32](5), CScalar[ci32](15))
        var result = logical_and[ci32](a, b)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input arrays must have the same shape for logical AND"
                    " operation."
                ),
                location="numojo.routines.logic.logical_and",
            )
        )
    var res: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](a.shape)
    for i in range(res.size):
        res.store(i, a.load(i) & b.load(i))
    return res^


def logical_or[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype.dtype == DType.bool or cdtype.dtype.is_integral()
):
    """
    Element-wise logical OR operation between two complex arrays.

    Args:
        a: First input complex array.
        b: Second input complex array.

    Raises:
        - NumojoError: If the input arrays do not have the same shape.

    Constraints:
        - Supports only boolean and integral complex data types.

    Returns:
        A complex array containing the result of the logical OR operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_or

        var a = nm.arange[ci32](CScalar[ci32](0), CScalar[ci32](10))
        var b = nm.arange[ci32](CScalar[ci32](5), CScalar[ci32](15))
        var result = logical_or[ci32](a, b)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input arrays must have the same shape for logical OR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_or",
            )
        )
    var res: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](a.shape)
    for i in range(res.size):
        res.store(i, a.load(i) | b.load(i))
    return res^


def logical_not[
    cdtype: ComplexDType
](a: ComplexNDArray[cdtype]) raises -> ComplexNDArray[cdtype] where (
    cdtype.dtype == DType.bool or cdtype.dtype.is_integral()
):
    """
    Element-wise logical NOT operation on a complex array.

    Args:
        a: Input complex array.

    Raises:
        - NumojoError: If the input array is not of a supported data type.

    Constraints:
        - Supports only boolean and integral complex data types.

    Returns:
        A complex array containing the result of the logical NOT operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_not

        var a = nm.arange[ci32](CScalar[ci32](0), CScalar[ci32](10))
        var result = logical_not[ci32](a)
        ```
    """
    var res: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](a.shape)
    for i in range(res.size):
        res.store(i, a.load(i).__invert__())
    return res^


def logical_xor[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype.dtype == DType.bool or cdtype.dtype.is_integral()
):
    """
    Element-wise logical XOR operation between two complex arrays.

    Args:
        a: First input complex array.
        b: Second input complex array.

    Raises:
        - NumojoError: If the input arrays do not have the same shape.

    Constraints:
        - Supports only boolean and integral complex data types.

    Returns:
        A complex array containing the result of the logical XOR operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_xor

        var a = nm.arange[ci32](CScalar[ci32](0), CScalar[ci32](10))
        var b = nm.arange[ci32](CScalar[ci32](5), CScalar[ci32](15))
        var result = logical_xor[ci32](a, b)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input arrays must have the same shape for logical XOR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_xor",
            )
        )
    var res: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](a.shape)
    for i in range(res.size):
        res.store(i, a.load(i) ^ b.load(i))
    return res^


# ===----------------------------------------------------------------------=== #
# Logical operations for NDArray
# ===----------------------------------------------------------------------=== #
