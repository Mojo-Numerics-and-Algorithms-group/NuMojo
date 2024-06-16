import math
import . _math_funcs as _mf

"""
implements comparison functions
"""


# ===------------------------------------------------------------------------===#
# Simple Elementwise Comparisons
# ===------------------------------------------------------------------------===#
fn greater[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: First NDArray to compare.
        tensor2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is greater than the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is greater than the corresponding element in y, and False otherwise.
    """
    return backend()._math_func_compare_2_tensors[dtype, SIMD.__gt__](
        tensor1, tensor2
    )


fn greater_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than or equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: First NDArray to compare.
        tensor2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is greater than or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is greater than or equal to the corresponding element in y, and False otherwise.
    """
    return backend()._math_func_compare_2_tensors[dtype, SIMD.__ge__](
        tensor1, tensor2
    )


fn less[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: First NDArray to compare.
        tensor2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is or equal to the corresponding element in y, and False otherwise.
    """
    return backend()._math_func_compare_2_tensors[dtype, SIMD.__lt__](
        tensor1, tensor2
    )


fn less_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are less than or equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: First NDArray to compare.
        tensor2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is less than or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is less than or equal to the corresponding element in y, and False otherwise.
    """
    return backend()._math_func_compare_2_tensors[dtype, SIMD.__le__](
        tensor1, tensor2
    )


fn equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: First NDArray to compare.
        tensor2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is equal to the corresponding element in y, and False otherwise.
    """
    return backend()._math_func_compare_2_tensors[dtype, SIMD.__eq__](
        tensor1, tensor2
    )


fn not_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are not equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: First NDArray to compare.
        tensor2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is not equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is not equal to the corresponding element in y, and False otherwise.
    """
    return backend()._math_func_compare_2_tensors[dtype, SIMD.__ne__](
        tensor1, tensor2
    )
