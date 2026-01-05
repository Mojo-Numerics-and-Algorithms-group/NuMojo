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
from numojo.core.matrix import Matrix, MatrixBase


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
    Determines whether two NDArrays are element-wise equal within a specified tolerance.

    This function compares each element of `a` and `b` and returns True if all corresponding elements satisfy the condition:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    Optionally, if `equal_nan` is True, NaN values at the same positions are considered equal.

    Parameters:
        dtype: The data type of the input NDArray.

    Args:
        a: First NDArray to compare.
        b: Second NDArray to compare.
        rtol: Relative tolerance (default: 1e-5). The maximum allowed difference, relative to the magnitude of `b`.
        atol: Absolute tolerance (default: 1e-8). The minimum absolute difference allowed.
        equal_nan: If True, NaNs in the same position are considered equal (default: False).

    Returns:
        True if all elements of `a` and `b` are equal within the specified tolerances, otherwise False.

    Example:
        ```mojo
        import numojo as nm
        from numojo.routines.logic.comparison import allclose
        var arr1 = nm.array[nm.f32]([1.0, 2.0, 3.0])
        var arr2 = nm.array[nm.f32]([1.0, 2.00001, 2.99999])
        print(allclose[nm.f32](arr1, arr2))  # Output: True
        ```
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
    Performs element-wise comparison of two NDArrays to determine if their values are equal within a specified tolerance.

    For each element pair (a_i, b_i), the result is True if:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    Optionally, if `equal_nan` is True, NaN values at the same positions are considered equal.

    Parameters:
        dtype: The data type of the input NDArray.

    Args:
        a: First NDArray to compare.
        b: Second NDArray to compare.
        rtol: Relative tolerance (default: 1e-5). The maximum allowed difference, relative to the magnitude of `b`.
        atol: Absolute tolerance (default: 1e-8). The minimum absolute difference allowed.
        equal_nan: If True, NaNs in the same position are considered equal (default: False).

    Returns:
        An NDArray of bools, where each element is True if the corresponding elements of `a` and `b` are equal within the specified tolerances, otherwise False.

    Example:
        ```mojo
        import numojo as nm
        from numojo.routines.logic.comparison import isclose
        var arr1 = nm.array[nm.f32]([1.0, 2.0, 3.0])
        var arr2 = nm.array[nm.f32]([1.0, 2.00001, 2.99999])
        print(isclose[nm.f32](arr1, arr2))  # Output: [True, True, True]
        ```
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


fn allclose[
    dtype: DType
](
    a: Matrix[dtype],
    b: Matrix[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Bool:
    """
    Determines whether two Matrix are element-wise equal within a specified tolerance.

    This function compares each element of `a` and `b` and returns True if all corresponding elements satisfy the condition:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    Optionally, if `equal_nan` is True, NaN values at the same positions are considered equal.

    Parameters:
        dtype: The data type of the input Matrix.

    Args:
        a: First Matrix to compare.
        b: Second Matrix to compare.
        rtol: Relative tolerance (default: 1e-5). The maximum allowed difference, relative to the magnitude of `b`.
        atol: Absolute tolerance (default: 1e-8). The minimum absolute difference allowed.
        equal_nan: If True, NaNs in the same position are considered equal (default: False).

    Returns:
        True if all elements of `a` and `b` are equal within the specified tolerances, otherwise False.

    Example:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.comparison import allclose
        var mat1 = Matrix.rand[f32]((2, 2))
        var mat2 = Matrix.rand[f32]((2, 2))
        print(allclose[f32](mat1, mat2))  # Output: True
        ```
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
    a: MatrixBase[dtype, **_],
    b: MatrixBase[dtype, **_],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Matrix[DType.bool]:
    """
    Performs element-wise comparison of two Matrix to determine if their values are equal within a specified tolerance.

    For each element pair (a_i, b_i), the result is True if:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    Optionally, if `equal_nan` is True, NaN values at the same positions are considered equal.

    Parameters:
        dtype: The data type of the input Matrix.

    Args:
        a: First Matrix to compare.
        b: Second Matrix to compare.
        rtol: Relative tolerance (default: 1e-5). The maximum allowed difference, relative to the magnitude of `b`.
        atol: Absolute tolerance (default: 1e-8). The minimum absolute difference allowed.
        equal_nan: If True, NaNs in the same position are considered equal (default: False).

    Returns:
        An NDArray of bools, where each element is True if the corresponding elements of `a` and `b` are equal within the specified tolerances, otherwise False.

    Example:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.comparison import isclose
        var mat1 = Matrix.rand[f32]((2, 2))
        var mat2 = Matrix.rand[f32]((2, 2))
        print(isclose[f32](mat1, mat2))
        ```
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

    var res: Matrix[DType.bool] = Matrix[DType.bool](a.shape)
    for i in range(a.size):
        val_a: Scalar[dtype] = a.load(i)
        val_b: Scalar[dtype] = b.load(i)
        if equal_nan and (math.isnan(val_a) and math.isnan(val_b)):
            res._store_idx(i, val=True)
            continue
        if abs(val_a - val_b) <= atol + rtol * abs(val_b):
            res._store_idx(i, val=True)
            continue
        else:
            res._store_idx(i, val=False)

    return res^


# TODO: define the allclose, isclose with correct behaviour for ComplexNDArray.


fn array_equal[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> Bool:
    """
    Checks if two NDArrays are equal element-wise and shape-wise.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        True if the two NDArrays are equal element-wise and shape-wise, False otherwise.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import array_equal

        var arr = nm.arange[i32](0, 10)
        var arr2 = nm.arange[i32](0, 10)
        print(array_equal[i32](arr, arr2))  # Output: True
        ```
    """
    if array1.shape != array2.shape:
        return False

    for i in range(array1.size):
        if array1.load(i) != array2.load(i):
            return False

    return True


# TODO: define array_equiv with correct broadcast semantics.
