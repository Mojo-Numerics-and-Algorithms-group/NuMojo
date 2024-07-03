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


@always_inline
fn calculate_block[
    M: Int, N: Int, K: Int, BLOCK_M: Int, BLOCK_N: Int, nelts: Int, dtype: DType
](
    res: NDArray[dtype],
    t1: NDArray[dtype],
    t2: NDArray[dtype],
    bm: Int,
    bn: Int,
) raises:
    # Compute tile
    var acc = stack_allocation[BLOCK_M * BLOCK_N, dtype]()
    memset_zero[dtype](acc, BLOCK_M * BLOCK_N)

    for k in range(K):
        # @unroll
        for m in range(BLOCK_M):

            @parameter
            fn inner_n[nelts: Int](n: Int):
                try:
                    acc.store[width=nelts](
                        m * BLOCK_N + n,
                        SIMD[dtype, nelts]
                        .splat(t1[(bm + m) * K + k])
                        .fma(
                            t2.load[width=nelts](k * N + (bn + n)),
                            acc.load[width=nelts](m * BLOCK_N + n),
                        ),
                    )
                except e:
                    print("Error", e)

            vectorize[inner_n, nelts](BLOCK_N)

    # Store tile
    for m in range(BLOCK_M):

        @parameter
        fn vec_store[nelts: Int](n: Int):
            var temp = acc.load[width=nelts](m * BLOCK_N + n)
            res.data.store[width=nelts]((bm + m) * N + (bn + n), val=temp)

        vectorize[vec_store, nelts](BLOCK_N)


@always_inline
fn dot[
    t10: Int, t11: Int, t21: Int, dtype: DType
](res: NDArray[dtype], t1: NDArray[dtype], t2: NDArray[dtype]) raises:
    alias M = t10  # t1[0]
    alias K = t11  # t1[1], t2[0]
    alias N = t21

    # simdwidthof[dtype]() = 8 for float32
    alias nelts = simdwidthof[dtype]()
    alias BLOCK_N = 8 * 2
    alias BLOCK_M = 6
    alias THREADS = 6  # num_logical_cores()

    alias BLOCK_N_REMAINDER = N % BLOCK_N
    alias BLOCK_M_REMAINDER = M % BLOCK_M

    @parameter
    fn bm_par(m_outer: Int):
        var bm = m_outer * BLOCK_M

        for n_outer in range(0, N // BLOCK_N):
            var bn = n_outer * BLOCK_N
            try:
                calculate_block[M, N, K, BLOCK_M, BLOCK_N, nelts](
                    res, t1, t2, bm, bn
                )
            except e:
                print("Error", e)

        # Handle the remainder of N
        @parameter
        if BLOCK_N_REMAINDER > 0:
            var bn = N - BLOCK_N_REMAINDER
            try:
                calculate_block[M, N, K, BLOCK_M, BLOCK_N_REMAINDER, nelts](
                    res, t1, t2, bm, bn
                )
            except e:
                print("Error", e)

    parallelize[bm_par](M // BLOCK_M, M // BLOCK_M)

    # Handle the remainder of M
    @parameter
    if BLOCK_M_REMAINDER > 0:
        var bm = M - BLOCK_M_REMAINDER

        for n_outer in range(0, N // BLOCK_N):
            var bn = n_outer * BLOCK_N

            calculate_block[M, N, K, BLOCK_M_REMAINDER, BLOCK_N, nelts](
                res, t1, t2, bm, bn
            )

        # Handle corner remainder
        @parameter
        if BLOCK_N_REMAINDER > 0:
            var bn = N - BLOCK_N_REMAINDER

            calculate_block[
                M, N, K, BLOCK_M_REMAINDER, BLOCK_N_REMAINDER, nelts
            ](res, t1, t2, bm, bn)
