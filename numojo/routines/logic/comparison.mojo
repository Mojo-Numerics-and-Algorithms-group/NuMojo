"""
Implements comparison math currently not using backend due to bool bitpacking issue
"""
# ===----------------------------------------------------------------------=== #
# Implements comparison functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #


import math

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray


# ===-------------------------------------a-----------------------------------===#
# Simple Element-wise Comparisons
# ===------------------------------------------------------------------------===#
fn greater[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are greater than values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is greater than the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is greater than the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.gt](array1, array2)


fn greater[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are greater than a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is greater than the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is greater than the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.gt](
        array1, scalar
    )


fn greater_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are greater than or equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is greater than or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is greater than or equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.ge](array1, array2)


fn greater_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are greater than or equal to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is greater than or equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is greater than or equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.ge](
        array1, scalar
    )


fn less[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is or equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.lt](array1, array2)


fn less[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is or equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is or equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.lt](
        array1, scalar
    )


fn less_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are less than or equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is less than or equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is less than or equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.le](array1, array2)


fn less_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are less than or equal to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is less than or equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is less than or equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.le](
        array1, scalar
    )


fn equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.eq](array1, array2)
    # if array1.shape != array2.shape:
    #         raise Error(
    #             "Shape Mismatch error shapes must match for this function"
    #         )
    # var result_array: NDArray[DType.bool] = NDArray[DType.bool](array1.shape)
    # for i in range(result_array.size()):
    #     result_array[i] = array1[i]==array2[i]
    # return result_array


fn equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are equal to a scalar.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.eq](
        array1, scalar
    )


fn not_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are not equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
    A NDArray containing True if the corresponding element in x is not equal to the corresponding element in y, otherwise False.

    An element of the result NDArray will be True if the corresponding element in x is not equal to the corresponding element in y, and False otherwise.
    """
    return backend().math_func_compare_2_arrays[dtype, SIMD.ne](array1, array2)
    # if array1.shape != array2.shape:
    #         raise Error(
    #             "Shape Mismatch error shapes must match for this function"
    #         )
    # var result_array: NDArray[DType.bool] = NDArray[DType.bool](array1.shape)
    # for i in range(result_array.size()):
    #     result_array[i] = array1[i]!=array2[i]
    # return result_array


fn not_equal[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise check of whether values in x are not equal to values in y.

    Parameters:
        dtype: The dtype of the input NDArray.
        backend: Sets utility function origin, defaults to `Vectorized.

    Args:
        array1: First NDArray to compare.
        scalar: Scalar to compare.

    Returns:
    A NDArray containing True if the element in x is not equal to the scalar, otherwise False.

    An element of the result NDArray will be True if the element in x is not equal to the scalar, and False otherwise.
    """
    return backend().math_func_compare_array_and_scalar[dtype, SIMD.ne](
        array1, scalar
    )


# TODO: Add backend to these functions.
fn allclose[
    dtype: DType
](
    a: NDArray[dtype],
    b: NDArray[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Bool:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        a: First NDArray to compare.
        b: Second NDArray to compare.
        rtol: The relative tolerance parameter (default is 1e-5).
        atol: The absolute tolerance parameter (default is 1e-8).
        equal_nan: If True, NaNs will be considered equal.

    Returns:
        True if the two arrays are equal within the given tolerance, False otherwise.
    """
    if a.shape != b.shape:
        raise Error(
            ShapeError(
                message=(
                    "Shape Mismatch error shapes must match for this function"
                ),
                location=(
                    "numojo.routines.logic.comparision.allclose(a: NDArray, b:"
                    " NDArray)"
                ),
            )
        )

    for i in range(a.size):
        val_a: Scalar[dtype] = a.load(i)
        val_b: Scalar[dtype] = b.load(i)
        if equal_nan and (math.isnan(val_a) and math.isnan(val_b)):
            continue
        if abs(val_a - val_b) <= atol + rtol * abs(val_b):
            continue
        else:
            return False

    return True


fn isclose[
    dtype: DType
](
    a: NDArray[dtype],
    b: NDArray[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> NDArray[DType.bool]:
    """
    Returns True if two scalars are equal within a tolerance.

    Parameters:
        dtype: The dtype of the input Scalar.

    Args:
        a: First Scalar to compare.
        b: Second Scalar to compare.
        rtol: The relative tolerance parameter (default is 1e-5).
        atol: The absolute tolerance parameter (default is 1e-8).
        equal_nan: If True, NaNs will be considered equal.

    Returns:
        True if the two scalars are equal within the given tolerance, False otherwise.
    """
    if a.shape != b.shape:
        raise Error(
            ShapeError(
                message=(
                    "Shape Mismatch error shapes must match for this function"
                ),
                location=(
                    "numojo.routines.logic.comparision.isclose(a: Scalar, b:"
                    " Scalar)"
                ),
            )
        )

    var res: NDArray[DType.bool] = NDArray[DType.bool](a.shape)
    for i in range(a.size):
        val_a: Scalar[dtype] = a.load(i)
        val_b: Scalar[dtype] = b.load(i)
        if equal_nan and (math.isnan(val_a) and math.isnan(val_b)):
            res.store(i, True)
            continue
        if abs(val_a - val_b) <= atol + rtol * abs(val_b):
            res.store(i, True)
            continue
        else:
            res.store(i, False)

    return res^
