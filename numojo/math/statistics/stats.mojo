"""
# ===----------------------------------------------------------------------=== #
# implements array stats function and supporting functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""
# from numojo.core.NDArray import NDArray
from ...core.ndarray import NDArray


fn sum(array: NDArray, axis: Int = 0) raises -> NDArray[array.dtype]:
    """
    Sum of array elements over a given axis.
    Args:
        array: NDArray.
        axis: The axis along which the sum is performed.
    Returns:
        An NDArray.

    """

    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.ndshape[i])
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
    var result: NDArray[array.dtype] = NDArray[array.dtype](
        NDArrayShape(result_shape)
    )

    for i in range(axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        result += arr_slice

    return result

fn sumall(array: NDArray) raises -> Scalar[array.dtype]:
    """
    Sum of all items in the array.
    Args:
        array: NDArray.
    Returns:
        Scalar.
    """

    var result = Scalar[array.dtype](0)
    for i in range(array.ndshape._size):
            result[0] += array.data[i]
    return result

fn prod(array: NDArray, axis: Int = 0) raises -> NDArray[array.dtype]:
    """
    Product of array elements over a given axis.
    Args:
        array: NDArray.
        axis: The axis along which the product is performed.
    Returns:
        An NDArray.

    """
    var ndim: Int = array.ndim
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(array.ndshape[i])
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

    slices[axis] = Slice(0, 1)
    result = array[slices]

    for i in range(1, axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = array[slices]
        result = result * arr_slice

    return result

fn prodall(array: NDArray) raises -> Scalar[array.dtype]:
    """
    Product of all items in the array.
    Args:
        array: NDArray.
    Returns:
        Scalar.
    """

    var result = Scalar[array.dtype](1)
    for i in range(array.ndshape._size):
            result[0] *= array.data[i]
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
    return sum(array, axis) / Scalar[array.dtype](array.ndshape[axis])
