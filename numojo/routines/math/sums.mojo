from sys import simdwidthof
from algorithm import parallelize, vectorize

from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros


fn sum[dtype: DType](A: NDArray[dtype]) -> Scalar[dtype]:
    """
    Returns sum of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
     [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
     [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32
    > print(nm.sum(A))
    3.5140917301177979
    ```

    Args:
        A: NDArray.

    Returns:
        Scalar.
    """

    alias width: Int = simdwidthof[dtype]()
    var res = Scalar[dtype](0)

    @parameter
    fn cal_vec[width: Int](i: Int):
        res += A._buf.load[width=width](i).reduce_add()

    vectorize[cal_vec, width](A.size)
    return res


fn sum[
    dtype: DType
](A: NDArray[dtype], owned axis: Int) raises -> NDArray[dtype]:
    """
    Returns sums of array elements over a given axis.

    Example:
    ```mojo
    import numojo as nm
    var A = nm.random.randn(100, 100)
    print(nm.sum(A, axis=0))
    ```

    Args:
        A: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """

    var ndim: Int = A.ndim
    if axis < 0:
        axis += ndim
    if (axis < 0) or (axis >= ndim):
        raise Error(
            String("axis {} greater than ndim of array {}").format(axis, ndim)
        )
    var result_shape: List[Int] = List[Int]()
    var size_of_axis: Int = A.shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(A.shape[i])
            slices.append(Slice(0, A.shape[i]))
        else:
            slices.append(Slice(0, 0))  # Temp value
    var result = zeros[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = A[slices]
        result += arr_slice

    return result


fn cumsum[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns cumsum of all items of an array.
    The array is flattened before cumsum.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.

    Returns:
        Cumsum of all items of an array.
    """

    if A.ndim == 1:
        var B = A
        for i in range(A.size - 1):
            B._buf[i + 1] += B._buf[i]
        return B^

    else:
        return cumsum(A.flatten(), axis=-1)


fn cumsum[
    dtype: DType
](owned A: NDArray[dtype], owned axis: Int) raises -> NDArray[dtype]:
    """
    Returns cumsum of array by axis.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.
        axis: Axis.

    Returns:
        Cumsum of array by axis.
    """

    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("axis {} greater than ndim of array {}").format(axis, A.ndim)
        )

    var I = NDArray[DType.index](Shape(A.size))
    var ptr = I._buf

    var _shape = A.shape._move_axis_to_end(axis)
    var _strides = A.strides._move_axis_to_end(axis)

    numojo.core.utility._traverse_buffer_according_to_shape_and_strides(
        ptr, _shape, _strides
    )

    for i in range(0, A.size, A.shape[axis]):
        for j in range(A.shape[axis] - 1):
            A._buf[int(I._buf[i + j + 1])] += A._buf[int(I._buf[i + j])]

    return A^
