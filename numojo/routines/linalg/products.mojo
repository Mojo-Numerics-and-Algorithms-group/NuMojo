"""
Matrix and vector products
"""
# ===----------------------------------------------------------------------=== #
# Matrix and vector products
# ===----------------------------------------------------------------------=== #


import math
from algorithm import parallelize, vectorize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from sys import simd_width_of
from memory import memcpy

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.core.matrix import Matrix, MatrixBase
from numojo.routines.creation import zeros
from numojo.routines.math.sums import sum


fn cross[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the cross product of two arrays.

    Parameters
        dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be of shape (3,).

    Returns:
        The cross product of two arrays.
    """

    if (array1.size == array2.size == 3) and (array1.ndim == array2.ndim == 1):
        var array3: NDArray[dtype] = NDArray[dtype](NDArrayShape(3))
        array3.store(
            0,
            (array1.load(1) * array2.load(2) - array1.load(2) * array2.load(1)),
        )
        array3.store(
            1,
            (array1.load(2) * array2.load(0) - array1.load(0) * array2.load(2)),
        )
        array3.store(
            2,
            (array1.load(0) * array2.load(1) - array1.load(1) * array2.load(0)),
        )
        return array3^
    else:
        raise Error(
            "resultross product is not supported for arrays of shape "
            + array1.shape.__str__()
            + " and "
            + array2.shape.__str__()
        )


# TODO: implement other cases for dot function
fn dot[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the dot product of two arrays.

    Parameters
        dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be 1 dimensional.

    Returns:
        The dot product of two arrays.
    """

    alias width = simd_width_of[dtype]()
    if array1.ndim == array2.ndim == 1:
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(array1.size))

        @parameter
        fn vectorized_dot[simd_width: Int](idx: Int) -> None:
            result._buf.ptr.store(
                idx,
                array1._buf.ptr.load[width=simd_width](idx)
                * array2._buf.ptr.load[width=simd_width](idx),
            )

        vectorize[vectorized_dot, width](array1.size)
        return result^
    else:
        raise Error(
            "resultross product is not supported for arrays of shape "
            + array1.shape.__str__()
            + " and "
            + array2.shape.__str__()
        )


# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# https://docs.modular.com/mojo/notebooks/Matmul
fn matmul_tiled_unrolled_parallelized[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Matrix multiplication vectorized, tiled, unrolled, and parallelized.
    """
    alias width = max(simd_width_of[dtype](), 16)
    var result: NDArray[dtype] = zeros[dtype](Shape(A.shape[0], B.shape[1]))
    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]

    @parameter
    fn calculate_A_rows(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[simd_width: Int](n: Int):
                    result._buf.ptr.store(
                        m * t2 + (n + x),
                        val=result._buf.ptr.load[width=simd_width](
                            m * t2 + (n + x)
                        )
                        + A._buf.ptr.load(m * t1 + k)
                        * B._buf.ptr.load[width=simd_width](k * t2 + (n + x)),
                    )

                alias unroll_factor = tile_x // width
                vectorize[
                    dot, width, size=tile_x, unroll_factor=unroll_factor
                ]()

        alias tile_size = 4
        tile[calc_tile, width * tile_size, tile_size](t1, t2)

    parallelize[calculate_A_rows](t0, t0)
    return result^


fn matmul_1darray[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication for 1-d arrays (inner dot).
    """

    var result = NDArray[dtype](Shape(1, 1))

    if A.ndim * B.ndim != 1:
        raise Error("The dimensions of the arrays should be 1.")
    elif A.size != B.size:
        raise Error(
            String(
                "matmul: a mismatch in core dimension 0: "
                "size {} is different from {}"
            ).format(A.size, B.size)
        )
    else:
        result._buf.ptr.init_pointee_copy(sum(A * B))

    return result^


fn matmul_2darray[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication for 2-d arrays (inner dot).

    Parameter:
        dtype: Data type.

    Args:
        A: First array.
        B: Second array.

    Return:
        A multiplied by B.

    Raises:
        When the shape does not match.

    Notes:
        The multiplication is vectorized and parallelized.

    References:
        [1] https://docs.modular.com/mojo/notebooks/Matmul.
        resultompared to the reference, we increases the size of
        the SIMD vector from the default width to 16. The purpose is to
        increase the performance via SIMD.
        This reduces the execution time by ~50 percent compared to
        `matmul_parallelized` and `matmul_tiled_unrolled_parallelized` for large
        matrices.
    """

    alias width = max(simd_width_of[dtype](), 16)

    if A.ndim * B.ndim == 1:
        return matmul_1darray(A, B)

    if (A.ndim == 1) and (A.size == B.shape[0]):
        var A_reshaped = A.reshape(Shape(1, A.shape[0]))
        var res = matmul_2darray(A_reshaped, B)
        return res.reshape(Shape(B.shape[1]))

    if (B.ndim == 1) and (A.shape[1] == B.size):
        var B_reshaped = B.reshape(Shape(B.shape[0], 1))
        var res = matmul_2darray(A, B_reshaped)
        return res.reshape(Shape(A.shape[0]))

    if (A.ndim == 1) or (B.ndim == 1):
        raise Error(
            String(
                "matmul: a mismatch in shapes: {} is different from {}"
            ).format(A.shape[-1], B.shape[0])
        )

    if A.shape[1] != B.shape[0]:
        raise Error(
            String(
                "matmul: a mismatch in shapes: {} is different from {}"
            ).format(A.shape[1], B.shape[0])
        )

    var result: NDArray[dtype] = zeros[dtype](Shape(A.shape[0], B.shape[1]))
    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]

    @parameter
    fn calculate_A_rows(m: Int):
        for k in range(t1):

            @parameter
            fn dot[simd_width: Int](n: Int):
                result._buf.ptr.store(
                    m * t2 + n,
                    val=result._buf.ptr.load[width=simd_width](m * t2 + n)
                    + A._buf.ptr.load[width=simd_width](m * t1 + k)
                    * B._buf.ptr.load[width=simd_width](k * t2 + n),
                )

            vectorize[dot, width](t2)

    parallelize[calculate_A_rows](t0, t0)

    return result^


fn matmul[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication for any dimensions.

    Parameter:
        dtype: Data type.

    Args:
        A: First array.
        B: Second array.

    Return:
        A multiplied by B.

    Raises:
        (1) The shapes of first n-2 dimensions do not match.
        (2) The shape of -2 dimension of first array does not match
        the shape of -1 dimension of the second array.

    Notes:\n
        When A and B are 1darray, it is equal to dot of vectors:
        `(i) @ (i) -> (1)`.\n
        When A and B are 2darray, it is equal to inner products of matrices:
        `(i,j) @ (j,k) -> (i,k)`.\n
        When A and B are more than 2d, it is equal to a stack of 2darrays:
        `(i,j,k) @ (i,k,l) -> (i,j,l)` and
        `(i,j,k,l) @ (i,j,l,m) -> (i,j,k,m)`.
    """

    if (A.ndim <= 2) and (B.ndim <= 2):
        return matmul_2darray(A, B)

    if A.ndim != B.ndim:
        raise Error(
            String("matmul: dimension {} is different from {}").format(
                A.ndim, B.ndim
            )
        )

    for i in range(A.ndim - 2):
        if A.shape[i] != B.shape[i]:
            raise Error(
                String("matmul: {}-th dimensions mismatch: {} vs {}").format(
                    A.shape[i], B.shape[i]
                )
            )

    if A.shape[-1] != B.shape[-2]:
        raise Error(
            String(
                "matmul: a mismatch in shapes: {} is different from {}"
            ).format(A.shape[-1], B.shape[-2])
        )

    var shape_as_list = List[Int]()
    for i in range(A.ndim - 2):
        shape_as_list.append(A.shape[i])
    shape_as_list.append(A.shape[-2])
    shape_as_list.append(B.shape[-1])

    var result = NDArray[dtype](Shape(shape_as_list))
    var A_sub_matrix = NDArray[dtype](Shape(A.shape[-2], A.shape[-1]))
    var B_sub_matrix = NDArray[dtype](Shape(B.shape[-2], B.shape[-1]))
    var result_sub_matrix = NDArray[dtype](
        Shape(result.shape[-2], result.shape[-1])
    )

    for i in range(result.size // result_sub_matrix.size):
        memcpy(
            dest=A_sub_matrix._buf.ptr,
            src=A._buf.ptr + (i * A_sub_matrix.size),
            count=A_sub_matrix.size,
        )
        memcpy(
            dest=B_sub_matrix._buf.ptr,
            src=B._buf.ptr + (i * B_sub_matrix.size),
            count=B_sub_matrix.size,
        )
        result_sub_matrix = matmul_2darray(A_sub_matrix, B_sub_matrix)
        memcpy(
            dest=result._buf.ptr + (i * result_sub_matrix.size),
            src=result_sub_matrix._buf.ptr,
            count=result_sub_matrix.size,
        )
    return result^


fn matmul[
    dtype: DType
](A: MatrixBase[dtype, **_], B: MatrixBase[dtype, **_]) raises -> Matrix[dtype]:
    """
    Matrix multiplication.

    Example:
    ```mojo
    from numojo import Matrix
    from numojo.routines.linalg import matmul
    var A = Matrix.rand(shape=(1000, 1000))
    var B = Matrix.rand(shape=(1000, 1000))
    var result = matmul(A, B)
    ```
    """

    alias width = max(simd_width_of[dtype](), 16)

    if A.shape[1] != B.shape[0]:
        raise Error(
            String("resultannot matmul {}x{} matrix with {}x{} matrix.").format(
                A.shape[0], A.shape[1], B.shape[0], B.shape[1]
            )
        )

    var result: Matrix[dtype]

    if A.flags.C_CONTIGUOUS and B.flags.C_CONTIGUOUS:
        result = Matrix.zeros[dtype](
            shape=(A.shape[0], B.shape[1]), order=B.order()
        )

        @parameter
        fn calculate_resultresult(m: Int):
            for k in range(A.shape[1]):

                @parameter
                fn dot[simd_width: Int](n: Int):
                    result._store[simd_width](
                        m,
                        n,
                        result._load[simd_width](m, n)
                        + A._load(m, k) * B._load[simd_width](k, n),
                    )

                vectorize[dot, width](B.shape[1])

        parallelize[calculate_resultresult](A.shape[0], A.shape[0])
    elif A.flags.F_CONTIGUOUS and B.flags.F_CONTIGUOUS:
        result = Matrix.zeros[dtype](
            shape=(A.shape[0], B.shape[1]), order=B.order()
        )

        @parameter
        fn calculate_FF(n: Int):
            for k in range(A.shape[1]):

                @parameter
                fn dot_F[simd_width: Int](m: Int):
                    result._store[simd_width](
                        m,
                        n,
                        result._load[simd_width](m, n)
                        + A._load[simd_width](m, k) * B._load(k, n),
                    )

                vectorize[dot_F, width](A.shape[0])

        parallelize[calculate_FF](B.shape[1], B.shape[1])
    elif A.flags.C_CONTIGUOUS and B.flags.F_CONTIGUOUS:
        result = Matrix.zeros[dtype](
            shape=(A.shape[0], B.shape[1]), order=B.order()
        )

        @parameter
        fn calculate_resultF(m: Int):
            for n in range(B.shape[1]):
                var sum: Scalar[dtype] = 0.0

                @parameter
                fn dot_product[simd_width: Int](k: Int):
                    sum += (
                        A._load[simd_width](m, k) * B._load[simd_width](k, n)
                    ).reduce_add()

                vectorize[dot_product, width](A.shape[1])
                result._store(m, n, sum)

        parallelize[calculate_resultF](A.shape[0], A.shape[0])

    else:
        result = matmul(A.reorder_layout(), B)

    return result^


fn matmul_naive[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Matrix multiplication with three nested loops.
    """
    var result: NDArray[dtype]
    if B.ndim == 1:
        result = zeros[dtype](NDArrayShape(A.shape[0]))
        for m in range(result.shape[0]):
            for k in range(A.shape[1]):
                result.store(m, val=result.load(m) + A.load(m, k) * B.load(k))
    elif B.ndim != 1:
        result = zeros[dtype](NDArrayShape(A.shape[0], B.shape[1]))
        for m in range(result.shape[0]):
            for k in range(A.shape[1]):
                for n in range(result.shape[1]):
                    result.store(
                        m,
                        n,
                        val=result.load(m, n) + A.load(m, k) * B.load(k, n),
                    )
    else:
        raise Error("Invalid shape for B")

    return result^
