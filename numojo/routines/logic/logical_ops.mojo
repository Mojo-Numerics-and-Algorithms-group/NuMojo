# ===----------------------------------------------------------------------=== #
# Logical Operations Module
# ===----------------------------------------------------------------------=== #
from numojo.core.error import ShapeError

# TODO: add `where` argument support to logical operations
# FIXME: Make all SIMD vectorized operations once bool bit-packing issue is resolved.
fn logical_and[dtype: DType](
    a: NDArray[dtype],
    b: NDArray[dtype]
) raises -> NDArray[DType.bool] where (dtype == DType.bool or dtype.is_integral()):
    """
    Element-wise logical AND operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical AND operation.
    """
    if a.shape != b.shape:
        raise Error(
            ShapeError(
                message="Input arrays must have the same shape for logical AND operation.",
                location="numojo.routines.logic.logical_and"
            )
        )
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](a.load(i) & b.load(i)))
    return res^

fn logical_or[dtype: DType](
    a: NDArray[dtype],
    b: NDArray[dtype]
) raises -> NDArray[DType.bool] where (dtype == DType.bool or dtype.is_integral()):
    """
    Element-wise logical OR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical OR operation.
    """
    if a.shape != b.shape:
        raise Error(
            ShapeError(
                message="Input arrays must have the same shape for logical OR operation.",
                location="numojo.routines.logic.logical_or"
            )
        )
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](a.load(i) | b.load(i)))
    return res^

fn logical_not[dtype: DType](
    a: NDArray[dtype]
) raises -> NDArray[DType.bool] where (dtype == DType.bool or dtype.is_integral()):
    """
    Element-wise logical NOT operation on an array.

    Args:
        a: Input array.

    Returns:
        An array containing the result of the logical NOT operation.
    """
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](~a.load(i)))
    return res^

fn logical_xor[dtype: DType](
    a: NDArray[dtype],
    b: NDArray[dtype]
) raises -> NDArray[DType.bool] where (dtype == DType.bool or dtype.is_integral()):
    """
    Element-wise logical XOR operation between two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        An array containing the result of the logical XOR operation.
    """
    if a.shape != b.shape:
        raise Error(
            ShapeError(
                message="Input arrays must have the same shape for logical XOR operation.",
                location="numojo.routines.logic.logical_xor"
            )
        )
    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(res.size):
        res.store(i, Scalar[DType.bool](a.load(i) ^ b.load(i)))
    return res^
