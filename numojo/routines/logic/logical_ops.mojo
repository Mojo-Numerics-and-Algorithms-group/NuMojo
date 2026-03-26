# ===----------------------------------------------------------------------=== #
# NuMojo: Logical ops
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Logical Operations Module (numojo.routines.logic.logical_ops)

This module implements element-wise logical operations for NDArray, ComplexNDArray, and Matrix types in the NuMojo library.
"""

from numojo.routines import HostExecutor
from numojo.core.error import NumojoError

# TODO: add `where` argument support to logical operations
# FIXME: Make all SIMD vectorized operations once bool bit-packing issue is resolved.
# TODO: Create backend for these operations.


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
    cdtype == ComplexDType.bool or cdtype.is_integral()
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
        var result = logical_and(a, b)
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
    cdtype == ComplexDType.bool or cdtype.is_integral()
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
        var result = logical_or(a, b)
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
    cdtype == ComplexDType.bool or cdtype.is_integral()
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
        var result = logical_not(a)
        ```
    """
    var res: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](a.shape)
    for i in range(res.size):
        res.store(i, ~a.load(i))
    return res^


def logical_xor[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype == ComplexDType.bool or cdtype.is_integral()
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
        var result = logical_xor(a, b)
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
# Logical operations for Matrix
# ===----------------------------------------------------------------------=== #


def logical_and[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical AND operation between two matrices.

    Args:
        a: First input matrix.
        b: Second input matrix.

    Raises:
        - NumojoError: If the input matrices do not have the same shape.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        A matrix containing the result of the logical AND operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_and

        var a = Matrix.rand[i32]((2, 5))
        var b = Matrix.rand[i32]((2, 5))
        var result = logical_and(a > 3, b < 10)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input matrices must have the same shape for logical AND"
                    " operation."
                ),
                location="numojo.routines.logic.logical_and",
            )
        )
    var res: Matrix[DType.bool] = Matrix[DType.bool](a.shape)
    for i in range(res.size):
        res._buf.store(i, Scalar[DType.bool](a.load(i) & b.load(i)))
    return res^


def logical_or[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical OR operation between two matrices.

    Args:
        a: First input matrix.
        b: Second input matrix.

    Raises:
        - NumojoError: If the input matrices do not have the same shape.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        A matrix containing the result of the logical OR operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_or

        var a = Matrix.rand[i32]((2, 5))
        var b = Matrix.rand[i32]((2, 5))
        var result = logical_or(a < 3, b > 10)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input matrices must have the same shape for logical OR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_or",
            )
        )
    var res: Matrix[DType.bool] = Matrix[DType.bool](a.shape)
    for i in range(res.size):
        res._buf.store(i, Scalar[DType.bool](a.load(i) | b.load(i)))
    return res^


def logical_not[
    dtype: DType
](a: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical NOT operation on a matrix.

    Args:
        a: Input matrix.

    Raises:
        - NumojoError: If the input matrix is not of a supported data type.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        A matrix containing the result of the logical NOT operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_not

        var a = Matrix.rand[i32]((2, 5))
        var result = logical_not(a < 5)
        ```
    """
    var res: Matrix[DType.bool] = Matrix[DType.bool](a.shape)
    for i in range(res.size):
        res._buf.store(i, Scalar[DType.bool](~a.load(i)))
    return res^


def logical_xor[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical XOR operation between two matrices.

    Args:
        a: First input matrix.
        b: Second input matrix.

    Raises:
        - NumojoError: If the input matrices do not have the same shape.

    Constraints:
        - Supports only boolean and integral data types.

    Returns:
        A matrix containing the result of the logical XOR operation.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_xor

        var a = Matrix.rand[i32]((2, 5))
        var b = Matrix.rand[i32]((2, 5))
        var result = logical_xor(a > 3, b < 10)
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Input matrices must have the same shape for logical XOR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_xor",
            )
        )
    var res: Matrix[DType.bool] = Matrix[DType.bool](a.shape)
    for i in range(res.size):
        res._buf.store(i, Scalar[DType.bool](a.load(i) ^ b.load(i)))
    return res^
