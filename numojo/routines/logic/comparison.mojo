# ===----------------------------------------------------------------------=== #
# NuMojo: Comparison
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Comparison routines (numojo.routines.logic.comparison)

Implements comparison math routines for NDArrays and Matrices.
"""

import std.math as math

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.core.error import NumojoError

# TODO: define the allclose, isclose with correct behaviour for ComplexNDArray.
# TODO: define array_equiv with correct broadcast semantics.

# ===------------------------------------------------------------------------===#
# Simple Element-wise Comparisons
# ===------------------------------------------------------------------------===#


def greater[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are greater than values in `array2`.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        An NDArray of bools, where each element is True if the corresponding element in `array1` is greater than the corresponding element in `array2`, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import greater

        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([0.5, 2.5, 2.0], shape=[3])
        print(greater[nm.f64](arr1, arr2))  # Output: [True, False, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.gt](array1, array2)


def greater[
    dtype: DType
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are greater than a scalar value.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: NDArray to compare.
        scalar: Scalar value to compare against.

    Returns:
        An NDArray of bools, where each element is True if the element in `array1` is greater than the scalar, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import greater

        var arr = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        print(greater[nm.f64](arr, 2.0))  # Output: [False, False, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.gt](array1, scalar)


def greater_equal[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are greater than or equal to values in `array2`.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        An NDArray of bools, where each element is True if the corresponding element in `array1` is greater than or equal to the corresponding element in `array2`, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import greater_equal

        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([0.5, 2.0, 4.0], shape=[3])
        print(greater_equal[nm.f64](arr1, arr2))  # Output: [True, True, False]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.ge](array1, array2)


def greater_equal[
    dtype: DType
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are greater than or equal to a scalar value.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: NDArray to compare.
        scalar: Scalar value to compare against.

    Returns:
        An NDArray of bools, where each element is True if the element in `array1` is greater than or equal to the scalar, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import greater_equal

        var arr = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        print(greater_equal[nm.f64](arr, 2.0))  # Output: [False, True, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.ge](array1, scalar)


def less[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are less than values in `array2`.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        An NDArray of bools, where each element is True if the corresponding element in `array1` is less than the corresponding element in `array2`, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import less

        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([0.5, 2.5, 2.0], shape=[3])
        print(less[nm.f64](arr1, arr2))  # Output: [False, True, False]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.lt](array1, array2)


def less[
    dtype: DType
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are less than a scalar value.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: NDArray to compare.
        scalar: Scalar value to compare against.

    Returns:
        An NDArray of bools, where each element is True if the element in `array1` is less than the scalar, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import less

        var arr = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        print(less[nm.f64](arr, 2.0))  # Output: [True, False, False]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.lt](array1, scalar)


def less_equal[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are less than or equal to values in `array2`.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        An NDArray of bools, where each element is True if the corresponding element in `array1` is less than or equal to the corresponding element in `array2`, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import less_equal

        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([0.5, 2.0, 4.0], shape=[3])
        print(less_equal[nm.f64](arr1, arr2))  # Output: [False, True, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.le](array1, array2)


def less_equal[
    dtype: DType
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are less than or equal to a scalar value.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: NDArray to compare.
        scalar: Scalar value to compare against.

    Returns:
        An NDArray of bools, where each element is True if the element in `array1` is less than or equal to the scalar, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import less_equal

        var arr = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        print(less_equal[nm.f64](arr, 2.0))  # Output: [True, True, False]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.le](array1, scalar)


def equal[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are equal to values in `array2`.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        An NDArray of bools, where each element is True if the corresponding element in `array1` is equal to the corresponding element in `array2`, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import equal

        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([1.0, 2.5, 3.0], shape=[3])
        print(equal[nm.f64](arr1, arr2))  # Output: [True, False, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.eq](array1, array2)


def equal[
    dtype: DType
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are equal to a scalar value.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: NDArray to compare.
        scalar: Scalar value to compare against.

    Returns:
        An NDArray of bools, where each element is True if the element in `array1` is equal to the scalar, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import equal

        var arr = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        print(equal[nm.f64](arr, 2.0))  # Output: [False, True, False]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.eq](array1, scalar)


def not_equal[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are not equal to values in `array2`.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        An NDArray of bools, where each element is True if the corresponding element in `array1` is not equal to the corresponding element in `array2`, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import not_equal

        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([1.0, 2.5, 2.0], shape=[3])
        print(not_equal[nm.f64](arr1, arr2))  # Output: [False, True, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.ne](array1, array2)


def not_equal[
    dtype: DType
](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
    """
    Performs element-wise comparison to check if values in `array1` are not equal to a scalar value.

    Parameters:
        dtype: The dtype of the input NDArray.

    Args:
        array1: NDArray to compare.
        scalar: Scalar value to compare against.

    Returns:
        An NDArray of bools, where each element is True if the element in `array1` is not equal to the scalar, otherwise False.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.logic.comparison import not_equal

        var arr = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        print(not_equal[nm.f64](arr, 2.0))  # Output: [True, False, True]
        ```
    """
    return HostExecutor.apply_binary_predicate[dtype, SIMD.ne](array1, scalar)


# ===------------------------------------------------------------------------===#
# Tolerance-based Comparisons
# ===------------------------------------------------------------------------===#


def allclose[
    dtype: DType
](
    a: NDArray[dtype],
    b: NDArray[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Bool:
    """
    Check if all elements of two NDArrays are equal within a given tolerance.

    For each element pair (a_i, b_i), this function returns True if:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    for all elements. If `equal_nan` is True, NaN values at the same position are considered equal.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: First array to compare.
        b: Second array to compare.
        rtol: Relative tolerance. Default is 1e-5.
        atol: Absolute tolerance. Default is 1e-8.
        equal_nan: If True, NaNs at the same position are considered equal. Default is False.

    Raises:
        NumojoError: If the shapes of `a` and `b` do not match.

    Returns:
        True if all elements are equal within the specified tolerances, otherwise False.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.logic.comparison import allclose
        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([1.0, 2.00001, 2.99999], shape=[3])
        print(allclose[nm.f64](arr1, arr2))  # Output: True.
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Shape Mismatch error shapes must match for this function"
                ),
                location=(
                    "numojo.routines.logic.comparison.allclose(a: NDArray, b:"
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


def isclose[
    dtype: DType
](
    a: NDArray[dtype],
    b: NDArray[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> NDArray[DType.bool]:
    """
    Perform element-wise comparison of two NDArrays to check if their values are equal within a given tolerance.

    For each element pair (a_i, b_i), the result is True if:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    If `equal_nan` is True, NaN values at the same position are considered equal.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: First array to compare.
        b: Second array to compare.
        rtol: Relative tolerance. Default is 1e-5.
        atol: Absolute tolerance. Default is 1e-8.
        equal_nan: If True, NaNs at the same position are considered equal. Default is False.

    Raises:
        NumojoError: If the shapes of `a` and `b` do not match.

    Returns:
        Array of bools, True where elements are equal within tolerance, otherwise False.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.logic.comparison import isclose
        var arr1 = nm.array[nm.f64]([1.0, 2.0, 3.0], shape=[3])
        var arr2 = nm.array[nm.f64]([1.0, 2.00001, 2.99999], shape=[3])
        print(isclose[nm.f64](arr1, arr2))  # Output: [True, True, True]
        ```
    """
    if a.shape != b.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Shape Mismatch error shapes must match for this function"
                ),
                location=(
                    "numojo.routines.logic.comparison.isclose(a: NDArray, b:"
                    " NDArray)"
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


def allclose[
    dtype: DType
](
    a: Matrix[dtype],
    b: Matrix[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Bool:
    """
    Check if all elements of two Matrix objects are equal within a given tolerance.

    For each element pair (a_i, b_i), this function returns True if:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    for all elements. If `equal_nan` is True, NaN values at the same position are considered equal.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: First matrix to compare.
        b: Second matrix to compare.
        rtol: Relative tolerance. Default is 1e-5.
        atol: Absolute tolerance. Default is 1e-8.
        equal_nan: If True, NaNs at the same position are considered equal. Default is False.

    Raises:
        NumojoError: If the shapes of `a` and `b` do not match.

    Returns:
        True if all elements are equal within the specified tolerances, otherwise False.

    Examples:
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
            NumojoError(
                category="shape",
                message=(
                    "Shape Mismatch error shapes must match for this function"
                ),
                location=(
                    "numojo.routines.logic.comparison.allclose(a: Matrix, b:"
                    " Matrix)"
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


def isclose[
    dtype: DType
](
    a: Matrix[dtype],
    b: Matrix[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Matrix[DType.bool]:
    """
    Perform element-wise comparison of two Matrix objects to check if their values are equal within a given tolerance.

    For each element pair (a_i, b_i), the result is True if:
        abs(a_i - b_i) <= atol + rtol * abs(b_i)
    If `equal_nan` is True, NaN values at the same position are considered equal.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: First matrix to compare.
        b: Second matrix to compare.
        rtol: Relative tolerance. Default is 1e-5.
        atol: Absolute tolerance. Default is 1e-8.
        equal_nan: If True, NaNs at the same position are considered equal. Default is False.

    Raises:
        NumojoError: If the shapes of `a` and `b` do not match.

    Returns:
        Matrix of bools, True where elements are equal within tolerance, otherwise False.

    Examples:
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
            NumojoError(
                category="shape",
                message=(
                    "Shape Mismatch error shapes must match for this function"
                ),
                location=(
                    "numojo.routines.logic.comparison.isclose(a: Matrix, b:"
                    " Matrix)"
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


# ===------------------------------------------------------------------------===#
# Exact Equality Comparisons
# ===------------------------------------------------------------------------===#


def array_equal[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> Bool:
    """
    Determine whether two NDArrays are exactly equal in both shape and element values.

    This function compares the shapes of `array1` and `array2`, and then checks each element for equality.
    The arrays are considered equal only if their shapes match and all corresponding elements are equal.

    Parameters:
        dtype: Data type of the array.

    Args:
        array1: First NDArray to compare.
        array2: Second NDArray to compare.

    Returns:
        True if both NDArrays have the same shape and all elements are equal; False otherwise.

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
