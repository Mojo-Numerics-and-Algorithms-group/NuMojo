"""
Statistics functions for NDArray
"""
# ===----------------------------------------------------------------------=== #
# implements array stats function and supporting functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #

# from numojo.core.NDArray import NDArray
from ...core.ndarray import NDArray
from .. import mul


fn sum(array: NDArray, axis: Int = 0) raises -> NDArray[array.dtype]:
    """Sum of array elements over a given axis.

    Args:
        array: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """

    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.shape[i])
    if axis > ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    var axis_size: Int = shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(shape[i])
            slices.append(Slice(0, shape[i]))
        else:
            slices.append(Slice(0, 0))
    var result: NDArray[array.dtype] = NDArray[array.dtype](
        NDArrayShape(result_shape)
    )
    for i in range(axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        result += arr_slice

    return result


fn sumall(array: NDArray) raises -> Scalar[array.dtype]:
    """Sum of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
    [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
    [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32
    > print(nm.math.stats.sumall(A))
    3.5140917301177979
    ```

    Args:
        array: NDArray.

    Returns:
        Scalar.
    """

    var result = Scalar[array.dtype](0)
    for i in range(array.shape.ndsize):
        result[0] += array._buf[i]
    return result


fn prod(array: NDArray, axis: Int = 0) raises -> NDArray[array.dtype]:
    """Product of array elements over a given axis.

    Args:
        array: NDArray.
        axis: The axis along which the product is performed.

    Returns:
        An NDArray.
    """
    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.shape[i])
    if axis > ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    var axis_size: Int = shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(shape[i])
            slices.append(Slice(0, shape[i]))
        else:
            slices.append(Slice(0, 0))

    slices[axis] = Slice(0, 1)
    var result: NDArray[array.dtype] = array[slices]

    for i in range(1, axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        result = mul[array.dtype](result, arr_slice)

    return result


fn prodall(array: NDArray) raises -> Scalar[array.dtype]:
    """Product of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
    [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
    [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32

    > print(nm.math.stats.prodall(A))
    6.1377261317829834e-07
    ```

    Args:
        array: NDArray.

    Returns:
        Scalar.
    """

    var result = Scalar[array.dtype](1)
    for i in range(array.shape.ndsize):
        result[0] *= array._buf[i]
    return result


fn mean(array: NDArray, axis: Int = 0) raises -> NDArray[array.dtype]:
    """
    Mean of array elements over a given axis.
    Args:
        array: NDArray.
        axis: The axis along which the mean is performed.
    Returns:
        An NDArray.

    """
    return sum(array, axis) / Scalar[array.dtype](array.shape[axis])


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
        sumall(array).cast[DType.float64]()
        / Int32(array.shape.ndsize).cast[DType.float64]()
    )


fn max[
    dtype: DType
](array: NDArray[dtype], axis: Int = 0) raises -> NDArray[dtype]:
    """Maximums of array elements over a given axis.

    Args:
        array: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """
    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.shape[i])
    if axis > ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    var axis_size: Int = shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(shape[i])
            slices.append(Slice(0, shape[i]))
        else:
            slices.append(Slice(0, 0))
    print(result_shape.__str__())

    slices[axis] = Slice(0, 1)

    var result: NDArray[dtype] = array[slices]
    for i in range(1, axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        var mask1 = greater(arr_slice, result)
        var mask2 = less(arr_slice, result)
        # Wherever result is less than the new slice it is set to zero
        # Wherever arr_slice is greater than the old result it is added to fill those zeros
        result = add(
            result * bool_to_numeric[dtype](mask2),
            arr_slice * bool_to_numeric[dtype](mask1),
        )

    return result


fn min[
    dtype: DType
](array: NDArray[dtype], axis: Int = 0) raises -> NDArray[dtype]:
    """Minumums of array elements over a given axis.

    Args:
        array: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """
    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.shape[i])
    if axis > ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result_shape: List[Int] = List[Int]()
    var axis_size: Int = shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(shape[i])
            slices.append(Slice(0, shape[i]))
        else:
            slices.append(Slice(0, 0))

    slices[axis] = Slice(0, 1)

    var result: NDArray[dtype] = array[slices]
    for i in range(1, axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        var mask1 = less(arr_slice, result)
        var mask2 = greater(arr_slice, result)
        # Wherever result is greater than the new slice it is set to zero
        # Wherever arr_slice is less than the old result it is added to fill those zeros
        result = add(
            result * bool_to_numeric[dtype](mask2),
            arr_slice * bool_to_numeric[dtype](mask1),
        )

    return result
