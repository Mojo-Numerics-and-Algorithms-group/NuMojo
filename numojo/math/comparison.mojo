"""
Implements comparison math currently not using backend due to bool bitpacking issue
"""
# ===----------------------------------------------------------------------=== #
# Implements comparison functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #


import math
import . math_funcs as _mf
from ..core.ndarray import NDArray


# ===-------------------------------------a-----------------------------------===#
# Simple Elementwise Comparisons
# ===------------------------------------------------------------------------===#
fn greater[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is greater than the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is greater than the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.__gt__](
        array1, array2
    )


fn greater[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is greater than the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is greater than the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.__gt__](
        array1, scalar
    )


fn greater_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than or equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is greater than or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is greater than or equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.__ge__](
        array1, array2
    )


fn greater_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are greater than or equal to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is greater than or equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is greater than or equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.__ge__](
        array1, scalar
    )


fn less[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is or equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.__lt__](
        array1, array2
    )


fn less[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is or equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is or equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.__lt__](
        array1, scalar
    )


fn less_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are less than or equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is less than or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is less than or equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.__le__](
        array1, array2
    )


fn less_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are less than or equal to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is less than or equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is less than or equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.__le__](
        array1, scalar
    )


fn equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.__eq__](
        array1, array2
    )
    # if array1.shape() != array2.shape():
    #         raise Error(
    #             "Shape Mismatch error shapes must match for this function"
    #         )
    # var result_array: NDArray[DType.bool] = NDArray[DType.bool](array1.shape())
    # for i in range(result_array.size()):
    #     result_array[i] = array1[i]==array2[i]
    # return result_array


fn equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are equal to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.__eq__](
        array1, scalar
    )


fn not_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are not equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is not equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is not equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.__ne__](
        array1, array2
    )
    # if array1.shape() != array2.shape():
    #         raise Error(
    #             "Shape Mismatch error shapes must match for this function"
    #         )
    # var result_array: NDArray[DType.bool] = NDArray[DType.bool](array1.shape())
    # for i in range(result_array.size()):
    #     result_array[i] = array1[i]!=array2[i]
    # return result_array


fn not_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs elementwise check of whether values in x are not equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is not equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is not equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.__ne__](
        array1, scalar
    )
