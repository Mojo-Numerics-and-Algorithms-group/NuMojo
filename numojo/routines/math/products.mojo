# from numojo.core.NDArray import NDArray
from numojo.core.ndarray import NDArray
from numojo.routines.math.arithmetic import mul


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
    for i in range(array.size):
        result[0] *= array._buf[i]
    return result


fn cumprod[
    dtype: DType = DType.float64
](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """Product of all items in an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The product of all items in the array as a SIMD Value of `dtype`.
    """

    var result: SIMD[dtype, 1] = SIMD[dtype, 1](1.0)
    alias width = simdwidthof[dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result *= simd_data.reduce_mul()

    vectorize[vectorize_sum, width](array.num_elements())
    return result
