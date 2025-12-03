# ===----------------------------------------------------------------------=== #
# Logical Operations Module
# ===----------------------------------------------------------------------=== #
from numojo.core.error import ShapeError


# TODO: add `where` argument support to logical operations
# FIXME: Make all SIMD vectorized operations once bool bit-packing issue is resolved.
# ===----------------------------------------------------------------------=== #
# NDArray operations
# ===----------------------------------------------------------------------=== #
fn logical_and[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical AND operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical AND operation.

    Raises:
        - ShapeError: If the input arrays do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
                message=(
                    "Input arrays must have the same shape for logical AND"
                    " operation."
                ),
                location="numojo.routines.logic.logical_and",
            )
        )
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](a.load(i) & b.load(i)))
    return res^


fn logical_or[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical OR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical OR operation.

    Raises:
        - ShapeError: If the input arrays do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
                message=(
                    "Input arrays must have the same shape for logical OR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_or",
            )
        )
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](a.load(i) | b.load(i)))
    return res^


fn logical_not[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical NOT operation on an array.

    Args:
        a: Input array.

    Returns:
        An array containing the result of the logical NOT operation.

    Raises:
        - ShapeError: If the input array is not of a supported data type.

    Notes:
        - Supports only boolean and integral data types.

    Example:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_not

        var a = nm.arange(0, 10)
        var result = logical_not(a < 5)
        ```
    """
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](~a.load(i)))
    return res^


fn logical_xor[
    dtype: DType
](a: NDArray[dtype], b: NDArray[dtype]) raises -> NDArray[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical XOR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical XOR operation.

    Raises:
        - ShapeError: If the input arrays do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
                message=(
                    "Input arrays must have the same shape for logical XOR"
                    " operation."
                ),
                location="numojo.routines.logic.logical_xor",
            )
        )
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](a.load(i) ^ b.load(i)))
    return res^


# ===----------------------------------------------------------------------=== #
# ComplexNDArray operations
# ===----------------------------------------------------------------------=== #
fn logical_and[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype == ComplexDType.bool or cdtype.is_integral()
):
    """
    Element-wise logical AND operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical AND operation.

    Raises:
        - ShapeError: If the input arrays do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
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


fn logical_or[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype == ComplexDType.bool or cdtype.is_integral()
):
    """
    Element-wise logical OR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical OR operation.

    Raises:
        - ShapeError: If the input arrays do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
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


fn logical_not[
    cdtype: ComplexDType
](a: ComplexNDArray[cdtype]) raises -> ComplexNDArray[cdtype] where (
    cdtype == ComplexDType.bool or cdtype.is_integral()
):
    """
    Element-wise logical NOT operation on an array.

    Args:
        a: Input array.

    Returns:
        An array containing the result of the logical NOT operation.

    Raises:
        - ShapeError: If the input array is not of a supported data type.

    Notes:
        - Supports only boolean and integral data types.

    Example:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_not

        var a = nm.arange(0, 10)
        var result = logical_not(a < 5)
        ```
    """
    var res: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](a.shape)
    for i in range(res.size):
        res.store(i, ~a.load(i))
    return res^


fn logical_xor[
    cdtype: ComplexDType
](
    a: ComplexNDArray[cdtype], b: ComplexNDArray[cdtype]
) raises -> ComplexNDArray[cdtype] where (
    cdtype == ComplexDType.bool or cdtype.is_integral()
):
    """
    Element-wise logical XOR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical XOR operation.

    Raises:
        - ShapeError: If the input arrays do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
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
# Matrix operations
# ===----------------------------------------------------------------------=== #
fn logical_and[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical AND operation between two matrices.

    Args:
        a: First input matrix.
        b: Second input matrix.

    Returns:
        A matrix containing the result of the logical AND operation.

    Raises:
        - ShapeError: If the input matrices do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
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


fn logical_or[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical OR operation between two matrices.

    Args:
        a: First input matrix.
        b: Second input matrix.

    Returns:
        A matrix containing the result of the logical OR operation.

    Raises:
        - ShapeError: If the input matrices do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
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


fn logical_not[
    dtype: DType
](
    a: Matrix[dtype],
) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical NOT operation on a matrix.

    Args:
        a: Input matrix.

    Returns:
        A matrix containing the result of the logical NOT operation.

    Raises:
        - ShapeError: If the input matrix is not of a supported data type.

    Notes:
        - Supports only boolean and integral data types.

    Example:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.logical_ops import logical_not
        var a = Matrix.rand[i32]((2, 5))
        var result = logical_not(a < 5)
        ```
    """
    var res: Matrix[DType.bool] = Matrix[DType.bool](a.shape)
    for i in range(res.size):
        res._buf.store(i, Scalar[DType.bool](a.load(i) | b.load(i)))
    return res^


fn logical_xor[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool] where (
    dtype == DType.bool or dtype.is_integral()
):
    """
    Element-wise logical XOR operation between two matrices.

    Args:
        a: First input matrix.
        b: Second input matrix.

    Returns:
        A matrix containing the result of the logical XOR operation.

    Raises:
        - ShapeError: If the input matrices do not have the same shape.

    Notes:
        - Supports only boolean and integral data types.

    Example:
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
            ShapeError(
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
