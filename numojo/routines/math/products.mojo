from algorithm import vectorize
from sys import simdwidthof

from numojo.core.ndarray import NDArray
from numojo.routines.creation import ones


fn prod[dtype: DType](A: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Returns products of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
    [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
    [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32

    > print(nm.prod(A))
    6.1377261317829834e-07
    ```

    Args:
        A: NDArray.

    Returns:
        Scalar.
    """

    alias width: Int = simdwidthof[dtype]()
    var res = Scalar[dtype](1)

    @parameter
    fn cal_vec[width: Int](i: Int):
        res *= A._buf.load[width=width](i).reduce_mul()

    vectorize[cal_vec, width](A.size)
    return res


fn prod[
    dtype: DType
](A: NDArray[dtype], owned axis: Int) raises -> NDArray[dtype]:
    """
    Returns products of array elements over a given axis.

    Args:
        A: NDArray.
        axis: The axis along which the product is performed.

    Returns:
        An NDArray.
    """

    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
        )

    var result_shape: List[Int] = List[Int]()
    var size_of_axis: Int = A.shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(A.ndim):
        if i != axis:
            result_shape.append(A.shape[i])
            slices.append(Slice(0, A.shape[i]))
        else:
            slices.append(Slice(0, 0))  # Temp value
    var result = ones[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = A[slices]
        result *= arr_slice

    return result


fn cumprod[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns cumprod of all items of an array.
    The array is flattened before cumprod.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.

    Returns:
        Cumprod of all items of an array.
    """

    if A.ndim == 1:
        var B = A
        for i in range(A.size - 1):
            B._buf[i + 1] *= B._buf[i]
        return B^

    else:
        return cumprod(A.flatten(), axis=-1)


fn cumprod[
    dtype: DType
](owned A: NDArray[dtype], owned axis: Int) raises -> NDArray[dtype]:
    """
    Returns cumprod of array by axis.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.
        axis: Axis.

    Returns:
        Cumprod of array by axis.
    """

    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
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
            A._buf[I._buf[i + j + 1]] *= A._buf[I._buf[i + j]]

    return A^
