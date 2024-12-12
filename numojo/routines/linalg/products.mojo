"""
Matrix and vector products
"""
# ===----------------------------------------------------------------------=== #
# Matrix and vector products
# ===----------------------------------------------------------------------=== #


import math
from algorithm import parallelize, vectorize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from sys import simdwidthof

import numojo.core._math_funcs as _mf
from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.routines.creation import zeros
from numojo.routines.math.sums import sumall


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
            (array1.get(1) * array2.get(2) - array1.get(2) * array2.get(1)),
        )
        array3.store(
            1,
            (array1.get(2) * array2.get(0) - array1.get(0) * array2.get(2)),
        )
        array3.store(
            2,
            (array1.get(0) * array2.get(1) - array1.get(1) * array2.get(0)),
        )
        return array3
    else:
        raise Error(
            "Cross product is not supported for arrays of shape "
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

    alias width = simdwidthof[dtype]()
    if array1.ndim == array2.ndim == 1:
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(array1.size))

        @parameter
        fn vectorized_dot[simd_width: Int](idx: Int) -> None:
            result.store[width=simd_width](
                idx,
                array1.load[width=simd_width](idx)
                * array2.load[width=simd_width](idx),
            )

        vectorize[vectorized_dot, width](array1.size)
        return result^
    else:
        raise Error(
            "Cross product is not supported for arrays of shape "
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
    alias width = max(simdwidthof[dtype](), 16)
    var C: NDArray[dtype] = zeros[dtype](
        Shape(A.shape.load_int(0), B.shape.load_int(1))
    )
    var t0 = A.shape.load_int(0)
    var t1 = A.shape.load_int(1)
    var t2 = B.shape.load_int(1)

    @parameter
    fn calculate_A_rows(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[simd_width: Int](n: Int):
                    C.store(
                        m * t2 + (n + x),
                        val=C.load[simd_width](m * t2 + (n + x))
                        + A.load(m * t1 + k)
                        * B.load[simd_width](k * t2 + (n + x)),
                    )

                alias unroll_factor = tile_x // width
                vectorize[
                    dot, width, size=tile_x, unroll_factor=unroll_factor
                ]()

        alias tile_size = 4
        tile[calc_tile, width * tile_size, tile_size](t1, t2)

    parallelize[calculate_A_rows](t0, t0)
    return C


fn matmul_1d[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """Array multiplication for 1-d arrays (inner dot)."""

    alias width = max(simdwidthof[dtype](), 16)

    try:
        if A.ndim * B.ndim != 1:
            raise Error("Dimension error!")
    except e:
        print(e)
        print(
            "The dimensions of the array should be 1.         A is of"
            " {A.ndim}-dimension. B is of {B.ndim}-dimension."
        )

    try:
        if A.size != B.size:
            raise Error("Size error!")
    except e:
        print(e)
        print("The sizes of the array should be identical.")

    var C: NDArray[dtype] = zeros[dtype](Shape(1, 1))

    C.store(0, val=sumall(A * B))

    return C^


fn matmul_parallelized[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """

    Matrix multiplication Vectorized and parallelized.

    Conduct `matmul` using `vectorize` and `parallelize`.

    Reference: https://docs.modular.com/mojo/notebooks/Matmul
    Compared to the reference, this function increases the size of
    the SIMD vector from the default width to 16. The purpose is to
    increase the performance via SIMD.
    The function reduces the execution time by ~50 percent compared to
    matmul_parallelized and matmul_tiled_unrolled_parallelized for large
    matrices.
    """

    alias width = max(simdwidthof[dtype](), 16)

    if A.ndim * B.ndim == 1:
        return matmul_1d(A, B)
    elif A.ndim == 1:
        A_reshaped = A
        A_reshaped.reshape(1, A_reshaped.shape[0])
        var res = A_reshaped @ B
        res.reshape(B.shape[1])
        return res
    elif B.ndim == 1:
        B_reshaped = B
        B_reshaped.reshape(B_reshaped.shape[0], 1)
        var res = A @ B_reshaped
        res.reshape(A.shape[0])
        return res

    var C: NDArray[dtype] = zeros[dtype](
        Shape(A.shape.load_int(0), B.shape.load_int(1))
    )
    var t0 = A.shape.load_int(0)
    var t1 = A.shape.load_int(1)
    var t2 = B.shape.load_int(1)

    @parameter
    fn calculate_A_rows(m: Int):
        for k in range(t1):

            @parameter
            fn dot[simd_width: Int](n: Int):
                C.store(
                    m * t2 + n,
                    val=C.load[simd_width](m * t2 + n)
                    + A.load(m * t1 + k) * B.load[simd_width](k * t2 + n),
                )

            vectorize[dot, width](t2)

    parallelize[calculate_A_rows](t0, t0)

    var _t0 = t0
    var _t1 = t1
    var _t2 = t2
    var _A = A
    var _B = B
    var _width = width

    return C^


fn matmul_naive[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Matrix multiplication with three nested loops.
    """
    var C: NDArray[dtype]
    if B.ndim == 1:
        C = zeros[dtype](NDArrayShape(A.shape[0]))
        for m in range(C.shape[0]):
            for k in range(A.shape[1]):
                C.store(m, val=C.load(m) + A.load(m, k) * B.load(k))
    elif B.ndim != 1:
        C = zeros[dtype](NDArrayShape(A.shape[0], B.shape[1]))
        for m in range(C.shape.load_int(0)):
            for k in range(A.shape.load_int(1)):
                for n in range(C.shape.load_int(1)):
                    C.store(
                        m, n, val=C.load(m, n) + A.load(m, k) * B.load(k, n)
                    )
    else:
        raise Error("Invalid shape for B")

    return C^
