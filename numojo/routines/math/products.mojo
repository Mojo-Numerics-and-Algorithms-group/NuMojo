from algorithm.functional import parallelize, vectorize
from sys import simd_width_of
from memory import UnsafePointer, memcpy, memset_zero

from numojo.core.ndarray import NDArray
from numojo.core.own_data import OwnData
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
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

    alias width: Int = simd_width_of[dtype]()
    var res = Scalar[dtype](1)

    @parameter
    fn cal_vec[width: Int](i: Int):
        res *= A._buf.ptr.load[width=width](i).reduce_mul()

    vectorize[cal_vec, width](A.size)
    return res


fn prod[
    dtype: DType
](A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
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
    var result: NDArray[dtype] = ones[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[axis] = Slice(i, i + 1)
        # TODO: modify slicing getter to avoid copy.
        var arr_slice: NDArray[dtype] = A._getitem_list_slices(slices.copy())
        result *= arr_slice

    return result^


fn prod[dtype: DType](A: Matrix[dtype, **_]) -> Scalar[dtype]:
    """
    Product of all items in the Matrix.

    Args:
        A: Matrix.
    """
    var res = Scalar[dtype](1)
    alias width: Int = simd_width_of[dtype]()

    @parameter
    fn cal_vec[width: Int](i: Int):
        res = res * A._buf.ptr.load[width=width](i).reduce_mul()

    vectorize[cal_vec, width](A.size)
    return res


fn prod[dtype: DType](A: Matrix[dtype, **_], axis: Int) raises -> Matrix[dtype]:
    """
    Product of items in a Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.prod(A, axis=0))
    print(mat.prod(A, axis=1))
    ```
    """

    alias width: Int = simd_width_of[dtype]()

    if axis == 0:
        var B = Matrix.ones[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int):
                B._store[width](
                    0, j, B._load[width](0, j) * A._load[width](i, j)
                )

            vectorize[cal_vec_sum, width](A.shape[1])

        return B^

    elif axis == 1:
        var B = Matrix.ones[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_vec[width: Int](j: Int):
                B._store(
                    i,
                    0,
                    B._load(i, 0) * A._load[width=width](i, j).reduce_mul(),
                )

            vectorize[cal_vec, width](A.shape[1])

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))


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
        var B = A.copy()
        for i in range(A.size - 1):
            B._buf.ptr[i + 1] *= B._buf.ptr[i]
        return B^

    else:
        return cumprod(A.flatten(), axis=-1)


fn cumprod[
    dtype: DType
](A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
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
    # TODO: reduce copies if possible
    var B: NDArray[dtype] = A.copy()
    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
        )

    var I = NDArray[DType.int](Shape(A.size))
    var ptr = I._buf.ptr

    var _shape = B.shape._move_axis_to_end(axis)
    var _strides = B.strides._move_axis_to_end(axis)

    numojo.core.utility._traverse_buffer_according_to_shape_and_strides(
        ptr, _shape, _strides
    )

    for i in range(0, B.size, B.shape[axis]):
        for j in range(B.shape[axis] - 1):
            B._buf.ptr[I._buf.ptr[i + j + 1]] *= B._buf.ptr[I._buf.ptr[i + j]]

    return B^


fn cumprod[
    dtype: DType
](A: Matrix[dtype, **_]) raises -> Matrix[dtype, OwnData]:
    """
    Cumprod of flattened matrix.

    Args:
        A: Matrix.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.cumprod(A))
    ```
    """
    alias width: Int = simd_width_of[dtype]()
    var result: Matrix[dtype] = Matrix.zeros[dtype](A.shape, "C")

    if A.flags.C_CONTIGUOUS:
        memcpy(dest=result._buf.ptr, src=A._buf.ptr, count=A.size)
    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                result[i, j] = A[i, j]

    for i in range(1, A.size):
        result._buf.ptr[i] *= result._buf.ptr[i - 1]

    result.resize(shape=(1, result.size))
    return result^


fn cumprod[
    dtype: DType
](A: Matrix[dtype, **_], axis: Int) raises -> Matrix[dtype, **_]:
    """
    Cumprod of Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.cumprod(A, axis=0))
    print(mat.cumprod(A, axis=1))
    ```
    """
    alias width: Int = simd_width_of[dtype]()
    var order: String = "C" if A.flags.C_CONTIGUOUS else "F"
    var result: Matrix[dtype] = Matrix.zeros[dtype](A.shape, order)

    if order == "C":
        memcpy(dest=result._buf.ptr, src=A._buf.ptr, count=A.size)
    else:
        for j in range(result.shape[1]):

            @parameter
            fn copy_col[width: Int](i: Int):
                result._store[width](i, j, A._load[width](i, j))

            vectorize[copy_col, width](A.shape[0])

    if axis == 0:
        if A.flags.C_CONTIGUOUS:
            for i in range(1, A.shape[0]):

                @parameter
                fn cal_vec_row[width: Int](j: Int):
                    result._store[width](
                        i,
                        j,
                        result._load[width](i - 1, j)
                        * result._load[width](i, j),
                    )

                vectorize[cal_vec_row, width](A.shape[1])
            return result^
        else:
            for j in range(A.shape[1]):
                for i in range(1, A.shape[0]):
                    result[i, j] = result[i - 1, j] * result[i, j]
            return result^

    elif axis == 1:
        if A.flags.C_CONTIGUOUS:
            for i in range(A.shape[0]):
                for j in range(1, A.shape[1]):
                    result[i, j] = result[i, j - 1] * result[i, j]
            return result^
        else:
            for j in range(1, A.shape[1]):

                @parameter
                fn cal_vec_column[width: Int](i: Int):
                    result._store[width](
                        i,
                        j,
                        result._load[width](i, j - 1)
                        * result._load[width](i, j),
                    )

                vectorize[cal_vec_column, width](A.shape[0])
            return result^
    else:
        raise Error(String("The axis can either be 1 or 0!"))
