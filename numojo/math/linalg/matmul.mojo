"""
# ===----------------------------------------------------------------------=== #
# implements matmul functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

import math
import .. _math_funcs as _mf
from ...core.ndarray import NDArray, NDArrayShape
from algorithm import parallelize, vectorize
from algorithm import Static2DTileUnitFunc as Tile2DFunc


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
    alias nelts = max(simdwidthof[dtype](), 16)
    var C: NDArray[dtype] = NDArray[dtype](
        A.ndshape.load_int(0), B.ndshape.load_int(1)
    )
    var t0 = A.ndshape.load_int(0)
    var t1 = A.ndshape.load_int(1)
    var t2 = B.ndshape.load_int(1)

    @parameter
    fn calculate_A_rows(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m * t2 + (n + x),
                        val=C.load[nelts](m * t2 + (n + x))
                        + A.load(m * t1 + k) * B.load[nelts](k * t2 + (n + x)),
                    )

                alias unroll_factor = tile_x // nelts
                vectorize[
                    dot, nelts, size=tile_x, unroll_factor=unroll_factor
                ]()

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](t1, t2)

    parallelize[calculate_A_rows](t0, t0)
    return C


fn matmul_parallelized[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """Conduct `matmul` using `vectorize` and `parallelize`.

    Reference: https://docs.modular.com/mojo/notebooks/Matmul
    Compared to the reference, this function increases the size of
    the SIMD vector from the default width to 16. The purpose is to
    increase the performance via SIMD.
    The function reduces the execution time by ~50 percent compared to
    matmul_parallelized and matmul_tiled_unrolled_parallelized for large
    matrices.
    """

    alias nelts = max(simdwidthof[dtype](), 16)

    var C: NDArray[dtype] = NDArray[dtype](
        A.ndshape.load_int(0), B.ndshape.load_int(1)
    )
    var t0 = A.ndshape.load_int(0)
    var t1 = A.ndshape.load_int(1)
    var t2 = B.ndshape.load_int(1)

    @parameter
    fn calculate_A_rows(m: Int):
        for k in range(t1):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store(
                    m * t2 + n,
                    val=C.load[nelts](m * t2 + n)
                    + A.load(m * t1 + k) * B.load[nelts](k * t2 + n),
                )

            vectorize[dot, nelts](t2)

    parallelize[calculate_A_rows](t0, t0)
    return C


fn matmul_naive[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    var C: NDArray[dtype] = NDArray[dtype](
        A.ndshape.load_int(0), B.ndshape.load_int(1)
    )
    for m in range(C.ndshape.load_int(0)):
        for k in range(A.ndshape.load_int(1)):
            for n in range(C.ndshape.load_int(1)):
                C.store(m, n, val=C.load(m, n) + A.load(m, k) * B.load(k, n))

    return C
