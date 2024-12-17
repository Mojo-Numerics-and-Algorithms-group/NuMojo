from sys import simdwidthof
from algorithm import parallelize, vectorize

from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros
from numojo.routines.math.arithmetic import mul


fn sum[dtype: DType](A: NDArray[dtype]) -> Scalar[dtype]:
    """Sum of all items in the array.

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

    var res = Scalar[dtype](0)
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn cal_vec[width: Int](i: Int):
        res = res + A._buf.load[width=width](i).reduce_add()

    vectorize[cal_vec, width](A.size)
    return res


fn sum[dtype: DType](A: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """Sum of array elements over a given axis.

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
    var shape: List[Int] = List[Int]()
    for i in range(ndim):
        shape.append(A.shape[i])
    if axis > ndim - 1:
        raise Error(
            String("axis {} greater than ndim of array {}").format(axis, ndim)
        )
    var result_shape: List[Int] = List[Int]()
    var axis_size: Int = shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(ndim):
        if i != axis:
            result_shape.append(shape[i])
            slices.append(Slice(0, shape[i]))
        else:
            slices.append(Slice(0, 0))
    var result = zeros[dtype](NDArrayShape(result_shape))
    for i in range(axis_size):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = A[slices]
        result += arr_slice

    return result


fn cumsum[
    dtype: DType = DType.float64
](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """Sum of all items of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The sum of all items in the array as a SIMD Value of `dtype`.
    """
    var result = Scalar[dtype]()
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result += simd_data.reduce_add()

    vectorize[vectorize_sum, width](array.num_elements())
    return result
