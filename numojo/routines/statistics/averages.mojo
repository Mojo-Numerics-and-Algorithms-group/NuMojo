"""
Averages and variances
"""
# ===----------------------------------------------------------------------=== #
# Averages and variances
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from math import sqrt

from numojo.core.ndarray import NDArray
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.core.utility import bool_to_numeric
from numojo.routines.logic.comparison import greater, less
from numojo.routines.math.arithmetic import add
from numojo.routines.sorting import binary_sort
from numojo.routines.math.sums import sum, cumsum
import numojo.routines.math.misc as misc


fn mean[
    dtype: DType
](array: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Mean of array elements over a given axis.
    Args:
        array: NDArray.
        axis: The axis along which the mean is performed.
    Returns:
        An NDArray.

    """
    return sum(array, axis) / Scalar[array.dtype](array.shape[axis])


fn mean[
    dtype_in: DType, //, dtype: DType = DType.float64
](a: NDArray[dtype_in]) raises -> Scalar[dtype]:
    """
    Calculate the arithmetic average of all items in the array.

    Args:
        a: NDArray.

    Returns:
        A scalar.

    """
    return sum(a).cast[dtype]() / a.size


fn mean[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Calculate the arithmetic average of all items in the Matrix.

    Args:
        A: Matrix.
    """

    return sum(A) / A.size


fn mean[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Calculate the arithmetic average of a Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.
    """

    if axis == 0:
        return sum(A, axis=0) / A.shape[0]
    elif axis == 1:
        return sum(A, axis=1) / A.shape[1]
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn meanall(array: NDArray) raises -> Float64:
    """Mean of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
    [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
    [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32

    > print(nm.math.stats.meanall(A))
    0.39045463667975533
    ```

    Args:
        array: NDArray.

    Returns:
        Float64.
    """

    return (
        sum(array).cast[DType.float64]()
        / Int32(array.size).cast[DType.float64]()
    )


fn cummean[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """Arithmatic mean of all items of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The mean of all of the member values of array as a SIMD Value of `dtype`.
    """
    return sum[dtype](array) / (array.num_elements())


fn mode[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """Mode of all items of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The mode of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_array: NDArray[dtype] = binary_sort[dtype](array)
    var max_count = 0
    var mode_value = sorted_array.item(0)
    var current_count = 1

    for i in range(1, array.num_elements()):
        if sorted_array[i] == sorted_array[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                mode_value = sorted_array.item(i - 1)
            current_count = 1

    if current_count > max_count:
        mode_value = sorted_array.item(array.num_elements() - 1)

    return mode_value


# * IMPLEMENT median high and low
fn median[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """Median value of all items of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The median of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_array = binary_sort[dtype](array)
    var n = array.num_elements()
    if n % 2 == 1:
        return sorted_array.item(n // 2)
    else:
        return (sorted_array.item(n // 2 - 1) + sorted_array.item(n // 2)) / 2


fn std[dtype: DType](A: Matrix[dtype], ddof: Int = 0) raises -> Scalar[dtype]:
    """
    Compute the standard deviation.

    Args:
        A: Matrix.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(String("ddof {ddof} should be smaller than size {A.size}"))

    return variance(A, ddof=ddof) ** 0.5


fn std[
    dtype: DType
](A: Matrix[dtype], axis: Int, ddof: Int = 0) raises -> Matrix[dtype]:
    """
    Compute the standard deviation along axis.

    Args:
        A: Matrix.
        axis: 0 or 1.
        ddof: Delta degree of freedom.
    """

    return variance(A, axis, ddof=ddof) ** 0.5


fn variance[
    dtype: DType
](A: Matrix[dtype], ddof: Int = 0) raises -> Scalar[dtype]:
    """
    Compute the variance.

    Args:
        A: Matrix.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(String("ddof {ddof} should be smaller than size {A.size}"))

    return sum((A - mean(A)) * (A - mean(A))) / (A.size - ddof)


fn variance[
    dtype: DType
](A: Matrix[dtype], axis: Int, ddof: Int = 0) raises -> Matrix[dtype]:
    """
    Compute the variance along axis.

    Args:
        A: Matrix.
        axis: 0 or 1.
        ddof: Delta degree of freedom.
    """

    if (ddof >= A.shape[0]) or (ddof >= A.shape[1]):
        raise Error(
            String(
                "ddof {ddof} should be smaller than size"
                " {A.shape[0]}x{A.shape[1]}"
            )
        )

    if axis == 0:
        return sum((A - mean(A, axis=0)) * (A - mean(A, axis=0)), axis=0) / (
            A.shape[0] - ddof
        )
    elif axis == 1:
        return sum((A - mean(A, axis=1)) * (A - mean(A, axis=1)), axis=1) / (
            A.shape[1] - ddof
        )
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn cumpvariance[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Population variance of a array.

    Parameters:
         dtype: The element type..

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    var mean_value: Scalar[dtype]
    if not mu:
        mean_value = cummean[dtype](array)
    else:
        mean_value = mu.value()

    var result = Scalar[dtype]()

    for i in range(array.num_elements()):
        result += (array.load(i) - mean_value) ** 2

    return sqrt(result / (array.num_elements()))


fn cumvariance[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Variance of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.

    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    var mean_value: Scalar[dtype]

    if not mu:
        mean_value = cummean[dtype](array)
    else:
        mean_value = mu.value()

    var result = Scalar[dtype]()
    for i in range(array.num_elements()):
        result += (array.load(i) - mean_value) ** 2

    return sqrt(result / (array.num_elements() - 1))


fn cumpstdev[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Population standard deviation of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.

    Returns:
        The standard deviation of all of the member values of array as a SIMD Value of `dtype`.
    """
    return sqrt(cumpvariance[dtype](array, mu))


fn cumstdev[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Standard deviation of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The standard deviation of all of the member values of array as a SIMD Value of `dtype`.
    """
    return sqrt(cumvariance[dtype](array, mu))
