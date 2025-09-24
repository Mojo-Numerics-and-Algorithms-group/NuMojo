from algorithm.functional import parallelize, vectorize
from sys import simd_width_of

from numojo.core.ndarray import NDArray
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
    var result = ones[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[axis] = Slice(i, i + 1)
        var arr_slice = A[slices]
        result *= arr_slice

    return result


fn prod[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
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


fn prod[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
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
        var B = A
        for i in range(A.size - 1):
            B._buf.ptr[i + 1] *= B._buf.ptr[i]
        return B^

    else:
        return cumprod(A.flatten(), axis=-1)


fn cumprod[
    dtype: DType
](var A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
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
    var ptr = I._buf.ptr

    var _shape = A.shape._move_axis_to_end(axis)
    var _strides = A.strides._move_axis_to_end(axis)

    numojo.core.utility._traverse_buffer_according_to_shape_and_strides(
        ptr, _shape, _strides
    )

    for i in range(0, A.size, A.shape[axis]):
        for j in range(A.shape[axis] - 1):
            A._buf.ptr[I._buf.ptr[i + j + 1]] *= A._buf.ptr[I._buf.ptr[i + j]]

    return A^


fn cumprod[dtype: DType](var A: Matrix[dtype]) -> Matrix[dtype]:
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
    var reorder = False
    if A.flags.F_CONTIGUOUS:
        reorder = True
        A = A.reorder_layout()

    A.resize(shape=(1, A.size))

    for i in range(1, A.size):
        A._buf.ptr[i] *= A._buf.ptr[i - 1]

    if reorder:
        A = A.reorder_layout()

    return A^


fn cumprod[
    dtype: DType
](var A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
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

    if axis == 0:
        if A.flags.C_CONTIGUOUS:
            for i in range(1, A.shape[0]):

                @parameter
                fn cal_vec_row[width: Int](j: Int):
                    A._store[width](
                        i, j, A._load[width](i - 1, j) * A._load[width](i, j)
                    )

                vectorize[cal_vec_row, width](A.shape[1])
            return A^
        else:
            for j in range(A.shape[1]):
                for i in range(1, A.shape[0]):
                    A[i, j] = A[i - 1, j] * A[i, j]
            return A^

    elif axis == 1:
        if A.flags.C_CONTIGUOUS:
            for i in range(A.shape[0]):
                for j in range(1, A.shape[1]):
                    A[i, j] = A[i, j - 1] * A[i, j]
            return A^
        else:
            for j in range(1, A.shape[1]):

                @parameter
                fn cal_vec_column[width: Int](i: Int):
                    A._store[width](
                        i, j, A._load[width](i, j - 1) * A._load[width](i, j)
                    )

                vectorize[cal_vec_column, width](A.shape[0])
            return A^
    else:
        raise Error(String("The axis can either be 1 or 0!"))
