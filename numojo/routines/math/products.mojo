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
    var result = ones[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = A[slices]
        result *= arr_slice

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
