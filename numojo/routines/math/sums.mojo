from sys import simd_width_of
from algorithm import parallelize, vectorize
from memory import UnsafePointer, memset_zero, memcpy

from numojo.core.ndarray import NDArray
from numojo.core.own_data import OwnData
from numojo.core.matrix import Matrix
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

    alias width: Int = simd_width_of[dtype]()
    var result: Scalar[dtype] = Scalar[dtype](0)

    @parameter
    fn cal_vec[width: Int](i: Int):
        result += A._buf.ptr.load[width=width](i).reduce_add()

    vectorize[cal_vec, width](A.size)
    return result


fn sum[dtype: DType](A: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Returns sums of array elements over a given axis.

    Example:
    ```mojo
    import numojo as nm
    var A = nm.random.randn(100, 100)
    print(nm.sum(A, axis=0))
    ```

    Raises:
        Error: If the axis is out of bound.
        Error: If the number of dimensions is 1.

    Args:
        A: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """

    var normalized_axis: Int = axis
    if normalized_axis < 0:
        normalized_axis += A.ndim

    if (normalized_axis < 0) or (normalized_axis >= A.ndim):
        raise Error(
            IndexError(
                message=String(
                    "Axis out of range: got {}; valid range is [0, {})."
                ).format(axis, A.ndim),
                suggestion=String(
                    "Use a valid axis in [0, {}) or a negative axis within"
                    " [-{}, -1]."
                ).format(A.ndim, A.ndim),
                location=String("routines.math.sums.sum(A, axis)"),
            )
        )
    if A.ndim == 1:
        raise Error(
            ShapeError(
                message=String("Cannot use axis with 1D array."),
                suggestion=String(
                    "Call `sum(A)` without axis, or reshape A to 2D or higher."
                ),
                location=String("routines.math.sums.sum(A, axis)"),
            )
        )

    var result_shape: List[Int] = List[Int]()
    var size_of_axis: Int = A.shape[normalized_axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(A.ndim):
        if i != normalized_axis:
            result_shape.append(A.shape[i])
            slices.append(Slice(0, A.shape[i]))
        else:
            slices.append(Slice(0, 0))  # Temp value
    var result: NDArray[dtype] = zeros[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[normalized_axis] = Slice(i, i + 1)
        var arr_slice: NDArray[dtype] = A._getitem_list_slices(slices.copy())
        result += arr_slice

    return result^


fn sum[dtype: DType](A: Matrix[dtype, **_]) -> Scalar[dtype]:
    """
    Sum up all items in the Matrix.

    Args:
        A: Matrix.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.sum(A))
    ```
    """
    var res = Scalar[dtype](0)
    alias width: Int = simd_width_of[dtype]()

    @parameter
    fn cal_vec[width: Int](i: Int):
        res = res + A._buf.ptr.load[width=width](i).reduce_add()

    vectorize[cal_vec, width](A.size)
    return res


fn sum[
    dtype: DType
](A: Matrix[dtype, **_], axis: Int) raises -> Matrix[dtype, OwnData]:
    """
    Sum up the items in a Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.sum(A, axis=0))
    print(mat.sum(A, axis=1))
    ```
    """

    alias width: Int = simd_width_of[dtype]()

    if axis == 0:
        var B = Matrix.zeros[dtype](shape=(1, A.shape[1]), order=A.order())

        if A.flags.F_CONTIGUOUS:

            @parameter
            fn calc_columns(j: Int):
                @parameter
                fn col_sum[width: Int](i: Int):
                    B._store(
                        0,
                        j,
                        B._load(0, j) + A._load[width=width](i, j).reduce_add(),
                    )

                vectorize[col_sum, width](A.shape[0])

            parallelize[calc_columns](A.shape[1], A.shape[1])
        else:
            for i in range(A.shape[0]):

                @parameter
                fn cal_vec_sum[width: Int](j: Int):
                    B._store[width](
                        0, j, B._load[width](0, j) + A._load[width](i, j)
                    )

                vectorize[cal_vec_sum, width](A.shape[1])

        return B^

    elif axis == 1:
        var B = Matrix.zeros[dtype](shape=(A.shape[0], 1), order=A.order())

        if A.flags.C_CONTIGUOUS:

            @parameter
            fn cal_rows(i: Int):
                @parameter
                fn cal_vec[width: Int](j: Int):
                    B._store(
                        i,
                        0,
                        B._load(i, 0) + A._load[width=width](i, j).reduce_add(),
                    )

                vectorize[cal_vec, width](A.shape[1])

            parallelize[cal_rows](A.shape[0], A.shape[0])
        else:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B._store(i, 0, B._load(i, 0) + A._load(i, j))

        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))


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
        var B = A.copy()
        for i in range(A.size - 1):
            B._buf.ptr[i + 1] += B._buf.ptr[i]
        return B^

    else:
        return cumsum(A.flatten(), axis=-1)


# Why do we do in inplace operation here?
fn cumsum[
    dtype: DType
](A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
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
            B._buf.ptr[Int(I._buf.ptr[i + j + 1])] += B._buf.ptr[
                Int(I._buf.ptr[i + j])
            ]

    return B^


fn cumsum[dtype: DType](A: Matrix[dtype, **_]) raises -> Matrix[dtype, OwnData]:
    """
    Cumsum of flattened matrix.

    Args:
        A: Matrix.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.cumsum(A))
    ```
    """
    var reorder = False
    var order = "C" if A.flags.C_CONTIGUOUS else "F"
    var result: Matrix[dtype, OwnData] = Matrix.zeros[dtype](A.shape, order)
    memcpy(dest=result._buf.ptr, src=A._buf.ptr, count=A.size)

    if A.flags.F_CONTIGUOUS:
        reorder = True
        result = result.reorder_layout()

    result.resize(shape=(1, A.size))

    for i in range(1, A.size):
        result._buf.ptr[i] += result._buf.ptr[i - 1]

    if reorder:
        result = result.reorder_layout()

    return result^


fn cumsum[
    dtype: DType
](A: Matrix[dtype, **_], axis: Int) raises -> Matrix[dtype, OwnData]:
    """
    Cumsum of Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import Matrix
    var A = Matrix.rand(shape=(100, 100))
    print(mat.cumsum(A, axis=0))
    print(mat.cumsum(A, axis=1))
    ```
    """

    alias width: Int = simd_width_of[dtype]()
    var order = "C" if A.flags.C_CONTIGUOUS else "F"
    var result: Matrix[dtype, OwnData] = Matrix.zeros[dtype](A.shape, order)
    memcpy(dest=result._buf.ptr, src=A._buf.ptr, count=A.size)

    if axis == 0:
        if result.flags.C_CONTIGUOUS:
            for i in range(1, A.shape[0]):

                @parameter
                fn cal_vec_sum_column[width: Int](j: Int):
                    result._store[width](
                        i, j, result._load[width](i - 1, j) + result._load[width](i, j)
                    )

                vectorize[cal_vec_sum_column, width](result.shape[1])
            return result^
        else:
            for j in range(A.shape[1]):
                for i in range(1, A.shape[0]):
                    result[i, j] = result[i - 1, j] + result[i, j]
            return result^

    elif axis == 1:
        if A.flags.C_CONTIGUOUS:
            for i in range(A.shape[0]):
                for j in range(1, A.shape[1]):
                    result[i, j] = result[i, j - 1] + result[i, j]
            return result^
        else:
            for j in range(1, A.shape[1]):

                @parameter
                fn cal_vec_sum_row[width: Int](i: Int):
                    result._store[width](
                        i, j, result._load[width](i, j - 1) + result._load[width](i, j)
                    )

                vectorize[cal_vec_sum_row, width](A.shape[0])
            return result^
    else:
        raise Error(String("The axis can either be 1 or 0!"))
