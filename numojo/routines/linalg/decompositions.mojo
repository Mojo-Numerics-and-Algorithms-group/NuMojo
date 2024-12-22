# ===----------------------------------------------------------------------=== #
# Decompositions
# ===----------------------------------------------------------------------=== #

from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros, eye, full
from algorithm import parallelize


fn lu_decomposition[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Perform LU (lower-upper) decomposition for matrix.

    Parameters:
        dtype: Data type of the upper and upper triangular matrices.

    Args:
        A: Input matrix for decoposition. It should be a row-major matrix.

    Returns:
        A tuple of the upper and lower triangular matrices.

    For efficiency, `dtype` of the output arrays will be the same as the input
    array. Thus, use `astype()` before passing the array to this function.

    Example:
    ```
    import numojo as nm
    fn main() raises:
        var arr = nm.NDArray[nm.f64]("[[1,2,3], [4,5,6], [7,8,9]]")
        var U: nm.NDArray
        var L: nm.NDArray
        L, U = nm.math.linalg.solver.lu_decomposition(arr)
        print(arr)
        print(L)
        print(U)
    ```
    ```console
    [[      1.0     2.0     3.0     ]
     [      4.0     5.0     6.0     ]
     [      7.0     8.0     9.0     ]]
    2-D array  Shape: [3, 3]  DType: float64
    [[      1.0     0.0     0.0     ]
     [      4.0     1.0     0.0     ]
     [      7.0     2.0     1.0     ]]
    2-D array  Shape: [3, 3]  DType: float64
    [[      1.0     2.0     3.0     ]
     [      0.0     -3.0    -6.0    ]
     [      0.0     0.0     0.0     ]]
    2-D array  Shape: [3, 3]  DType: float64
    ```

    Further reading:
        Linear Algebra And Its Applications, fourth edition, Gilbert Strang
        https://en.wikipedia.org/wiki/LU_decomposition
        https://www.scicoding.com/how-to-calculate-lu-decomposition-in-python/
        https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html

    TODO: Optimize the speed.

    """

    # Check whether the dimension is 2
    if A.ndim != 2:
        raise ("The array is not 2-dimensional!")

    # Check whether the matrix is square
    var shape_of_array = A.shape
    if shape_of_array[0] != shape_of_array[1]:
        raise ("The matrix is not square!")
    var n = shape_of_array[0]

    # Check whether the matrix is singular
    # if singular:
    #     raise("The matrix is singular!")

    # Change dtype of array to defined dtype
    # var A = array.astype[dtype]()

    # Initiate upper and lower triangular matrices
    var U = full[dtype](shape=shape_of_array, fill_value=SIMD[dtype, 1](0))
    var L = full[dtype](shape=shape_of_array, fill_value=SIMD[dtype, 1](0))

    # Fill in L and U
    # @parameter
    # fn calculate(i: Int):
    for i in range(0, n):
        for j in range(i, n):
            # Fill in L
            if i == j:
                L.store[width=1](i * n + i, 1)
            else:
                var sum_of_products_for_L: Scalar[dtype] = 0
                for k in range(0, i):
                    sum_of_products_for_L += L.load(j * n + k) * U.load(
                        k * n + i
                    )
                L.store[width=1](
                    j * n + i,
                    (A.load(j * n + i) - sum_of_products_for_L)
                    / U.load(i * n + i),
                )

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L.load(i * n + k) * U.load(k * n + j)
            U.store[width=1](
                i * n + j, A.load(i * n + j) - sum_of_products_for_U
            )

    # parallelize[calculate](n, n)

    return L, U
