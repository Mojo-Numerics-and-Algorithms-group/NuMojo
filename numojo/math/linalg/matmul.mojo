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

# fn matmul[
#     dtype: DType
# ](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]:

#     var result = NDArray[dtype](array1.shape(), 0.0)
#     alias opt_nelts = simdwidthof[dtype]()

#     @parameter
#     fn calc_row(m: Int):
#         for k in range(self.info.shape[1]):
#             @parameter
#             fn dot[nelts : Int](n : Int):
#                 result.store[nelts](m, n, val=result.load[nelts](m,n)
#                     + self.load[nelts](m,k) * other.load[nelts](k,n))
#             vectorize[dot, opt_nelts](other.info.shape[1])
#     parallelize[calc_row](self.info.shape[0], self.info.shape[0])
#     return result


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
    alias nelts = simdwidthof[dtype]()
    var C: NDArray[dtype] = NDArray[dtype](
        A.ndshape._shape[0], B.ndshape._shape[1]
    )
    # print(C.info.shape[0], "x", C.info.shape[1])

    @parameter
    fn calculate_A_rows(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        val=C.load[nelts](m, n + x)
                        + A.load(m, k) * B.load[nelts](k, n + x),
                    )

                alias unroll_factor = tile_x // nelts
                vectorize[
                    dot, nelts, size=tile_x, unroll_factor=unroll_factor
                ]()

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](
            A.ndshape._shape[1], C.ndshape._shape[1]
        )

    parallelize[calculate_A_rows](C.ndshape._shape[0], C.ndshape._shape[0])
    return C


fn matmul[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    alias nelts = simdwidthof[dtype]()

    var C: NDArray[dtype] = NDArray[dtype](
        A.ndshape._shape[0], B.ndshape._shape[1]
    )

    # print(C.info.shape[0], "x", C.info.shape[1])
    @parameter
    fn calc_row(m: Int):
        for k in range(A.ndshape._shape[1]):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m,
                    n,
                    val=C.load[nelts](m, n)
                    + A.load(m, k) * B.load[nelts](k, n),
                )

            vectorize[dot, nelts](B.ndshape._shape[1])

    parallelize[calc_row](C.ndshape._shape[0])

    # var C: NDArray[dtype] = NDArray[dtype](A.info.shape[0], B.info.shape[1])
    # for m in range(C.info.shape[0]):
    #     for k in range(A.info.shape[1]):
    #         @parameter
    #         fn dot[nelts: Int](n: Int):
    #             C.store(m,n,
    #                 val=C.load[nelts](m,n) + A.load(m,k) * B.load[nelts](k,n))
    #         vectorize[dot, nelts](C.info.shape[1])

    return C


fn matmul_parallelized[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    alias nelts = simdwidthof[dtype]()

    var C: NDArray[dtype] = NDArray[dtype](A.ndshape[0], B.ndshape[1])
    print(C.ndshape[0], "x", C.ndshape[1])

    @parameter
    fn calculate_A_rows(m: Int):
        for k in range(A.ndshape[1]):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store(
                    m,
                    n,
                    val=C.load[nelts](m, n)
                    + A.load(m, k) * B.load[nelts](k, n),
                )

            vectorize[dot, nelts](C.ndshape[1])

    parallelize[calculate_A_rows](C.ndshape[0], C.ndshape[0])
    return C


fn matmul_naive[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    var C: NDArray[dtype] = NDArray[dtype](A.info.shape[0], B.info.shape[1])

    for m in range(C.info.shape[0]):
        for k in range(A.info.shape[1]):
            for n in range(C.info.shape[1]):
                C.store(m, n, val=C.load(m, n) + A.load(m, k) * B.load(k, n))

    # for m in range(C.info.shape[0]):
    # for k in range(A.info.shape[1]):
    #     for n in range(C.info.shape[1]):
    #         C.__setitem__(List[Int](m,n), val=C.__getitem__(m,n) + A.__getitem__(m,k) * B.__getitem__(k,n))

    return C
