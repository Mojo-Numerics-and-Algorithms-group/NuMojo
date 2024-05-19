import math
import . _math_funcs as _mf
from tensor import Tensor

"""
implements comparison functions
"""


# ===------------------------------------------------------------------------===#
# Simple Elementwise Comparisons
# ===------------------------------------------------------------------------===#
fn greater[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than values in y.

    Parameters:
        dtype: The dtype of the input Tensor.

    Args:
        tensor1: First Tensor to compare.
        tensor2: Second Tensor to compare.

    Returns:
    A Tensor containing True if the corresponding element in x is greater than the corresponding element in y, otherwise False.

    An element of the result Tensor will be True if the corresponding element in x is greater than the corresponding element in y, and False otherwise.
    """
    return _mf._math_func_compare_2_tensors[dtype, math.greater](
        tensor1, tensor2
    )


fn greater_equal[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than or equal to values in y.

    Parameters:
        dtype: The dtype of the input Tensor.

    Args:
        tensor1: First Tensor to compare.
        tensor2: Second Tensor to compare.

    Returns:
    A Tensor containing True if the corresponding element in x is greater than or equal to the corresponding element in y, otherwise False.

    An element of the result Tensor will be True if the corresponding element in x is greater than or equal to the corresponding element in y, and False otherwise.
    """
    return _mf._math_func_compare_2_tensors[dtype, math.greater_equal](
        tensor1, tensor2
    )


fn less[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """
    Performs elementwise check of whether values in x are to values in y.

    Parameters:
        dtype: The dtype of the input Tensor.

    Args:
        tensor1: First Tensor to compare.
        tensor2: Second Tensor to compare.

    Returns:
    A Tensor containing True if the corresponding element in x is or equal to the corresponding element in y, otherwise False.

    An element of the result Tensor will be True if the corresponding element in x is or equal to the corresponding element in y, and False otherwise.
    """
    return _mf._math_func_compare_2_tensors[dtype, math.less](tensor1, tensor2)


fn less_equal[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """
    Performs elementwise check of whether values in x are less than or equal to values in y.

    Parameters:
        dtype: The dtype of the input Tensor.

    Args:
        tensor1: First Tensor to compare.
        tensor2: Second Tensor to compare.

    Returns:
    A Tensor containing True if the corresponding element in x is less than or equal to the corresponding element in y, otherwise False.

    An element of the result Tensor will be True if the corresponding element in x is less than or equal to the corresponding element in y, and False otherwise.
    """
    return _mf._math_func_compare_2_tensors[dtype, math.less_equal](
        tensor1, tensor2
    )


fn equal[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """
    Performs elementwise check of whether values in x are equal to values in y.

    Parameters:
        dtype: The dtype of the input Tensor.

    Args:
        tensor1: First Tensor to compare.
        tensor2: Second Tensor to compare.

    Returns:
    A Tensor containing True if the corresponding element in x is equal to the corresponding element in y, otherwise False.

    An element of the result Tensor will be True if the corresponding element in x is equal to the corresponding element in y, and False otherwise.
    """
    return _mf._math_func_compare_2_tensors[dtype, math.equal](tensor1, tensor2)


fn not_equal[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """
    Performs elementwise check of whether values in x are not equal to values in y.

    Parameters:
        dtype: The dtype of the input Tensor.

    Args:
        tensor1: First Tensor to compare.
        tensor2: Second Tensor to compare.

    Returns:
    A Tensor containing True if the corresponding element in x is not equal to the corresponding element in y, otherwise False.

    An element of the result Tensor will be True if the corresponding element in x is not equal to the corresponding element in y, and False otherwise.
    """
    return _mf._math_func_compare_2_tensors[dtype, math.not_equal](
        tensor1, tensor2
    )
