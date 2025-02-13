# ===----------------------------------------------------------------------=== #
# Decompositions
# ===----------------------------------------------------------------------=== #

from algorithm import parallelize
import math as builtin_math

from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.routines.creation import zeros, eye, full


fn compute_householder[
    dtype: DType
](
    mut H: Matrix[dtype], mut R: Matrix[dtype], row: Int, column: Int
) raises -> None:
    var sqrt2: SIMD[dtype, 1] = 1.4142135623730951
    var rRows = R.shape[0]

    for i in range(row, rRows):
        var val = R._load(i, column)
        H._store(i, column, val)
        R._store(i, column, 0.0)

    var norm: Scalar[dtype] = 0.0
    for i in range(rRows):
        norm += H._load(i, column) ** 2
    norm = builtin_math.sqrt(norm)
    if row == rRows - 1 or norm == 0:
        first_element = H._load(row, column)
        R._store(row, column, -first_element)
        H._store(row, column, sqrt2)
        return

    scale = 1.0 / norm
    if H._load(row, column) < 0:
        scale = -scale

    R._store(row, column, -1 / scale)

    for i in range(row, rRows):
        H._store(i, column, H._load(i, column) * scale)

    increment = H._load(row, column) + 1.0
    H._store(row, column, increment)

    s = builtin_math.sqrt(1.0 / increment)

    for i in range(row, rRows):
        H._store(i, column, H._load(i, column) * s)


fn compute_qr[
    dtype: DType
](
    mut H: Matrix[dtype],
    work_index: Int,
    mut A: Matrix[dtype],
    row_start: Int,
    column_start: Int,
) raises -> None:
    var aRows = A.shape[0]
    var aCols = A.shape[1]

    for j in range(column_start, aCols):
        var dot: SIMD[dtype, 1] = 0.0
        for i in range(row_start, aRows):
            dot += H._load(i, work_index) * A._load(i, j)
        for i in range(row_start, aRows):
            val = A._load(i, j) - H._load(i, work_index) * dot
            A._store(i, j, val)


fn lu_decomposition[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Perform LU (lower-upper) decomposition for array.

    Parameters:
        dtype: Data type of the upper and upper triangular matrices.

    Args:
        A: Input matrix for decomposition. It should be a row-major matrix.

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
        L, U = nm.linalg.lu_decomposition(arr)
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

    Further readings:
    - Linear Algebra And Its Applications, fourth edition, Gilbert Strang
    - https://en.wikipedia.org/wiki/LU_decomposition
    - https://www.scicoding.com/how-to-calculate-lu-decomposition-in-python/
    - https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html.
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


fn lu_decomposition[
    dtype: DType
](A: Matrix[dtype]) raises -> Tuple[Matrix[dtype], Matrix[dtype]]:
    """
    Perform LU (lower-upper) decomposition for matrix.
    """
    # Check whether the matrix is square
    if A.shape[0] != A.shape[1]:
        raise Error(
            String("{}x{} matrix is not square.").format(A.shape[0], A.shape[1])
        )

    var n = A.shape[0]

    # Initiate upper and lower triangular matrices
    var U = Matrix.full[dtype](shape=(n, n))
    var L = Matrix.full[dtype](shape=(n, n))

    # Fill in L and U
    for i in range(0, n):
        for j in range(i, n):
            # Fill in L
            if i == j:
                L._store(i, i, 1)
            else:
                var sum_of_products_for_L: Scalar[dtype] = 0
                for k in range(0, i):
                    sum_of_products_for_L += L._load(j, k) * U._load(k, i)
                L._store(
                    j,
                    i,
                    (A._load(j, i) - sum_of_products_for_L) / U._load(i, i),
                )

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L._load(i, k) * U._load(k, j)
            U._store(i, j, A._load(i, j) - sum_of_products_for_U)

    return L, U


fn partial_pivoting[
    dtype: DType
](owned A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype], Int]:
    """
    Perform partial pivoting for a square matrix.

    Args:
        A: 2-d square array.

    Returns:
        Pivoted array.
        The permutation matrix.
        The number of exchanges.
    """

    if A.ndim != 2:
        raise Error(String("Array must be 2d."))
    if A.shape[0] != A.shape[1]:
        raise Error(String("Matrix is not square."))

    var n = A.shape[0]
    var P = identity[dtype](n)
    var s: Int = 0  # Number of exchanges, for determinant

    for col in range(n):
        var max_p = abs(A.item(col, col))
        var max_p_row = col
        for row in range(col + 1, n):
            if abs(A.item(row, col)) > max_p:
                max_p = abs(A.item(row, col))
                max_p_row = row

        for i in range(n):
            # A[col], A[max_p_row] = A[max_p_row], A[col]
            # P[col], P[max_p_row] = P[max_p_row], P[col]
            var temp = A.item(max_p_row, i)
            A[Item(max_p_row, i)] = A.item(col, i)
            A[Item(col, i)] = temp

            temp = P.item(max_p_row, i)
            P[Item(max_p_row, i)] = P.item(col, i)
            P[Item(col, i)] = temp

        if max_p_row != col:
            s = s + 1

    return Tuple(A^, P^, s)


fn partial_pivoting[
    dtype: DType
](owned A: Matrix[dtype]) raises -> Tuple[Matrix[dtype], Matrix[dtype], Int]:
    """
    Perform partial pivoting for matrix.
    """
    var n = A.shape[0]
    var P = Matrix.identity[dtype](n)
    var s: Int = 0  # Number of exchanges, for determinant
    for col in range(n):
        var max_p = abs(A[col, col])
        var max_p_row = col
        for row in range(col + 1, n):
            if abs(A[row, col]) > max_p:
                max_p = abs(A[row, col])
                max_p_row = row
        A[col], A[max_p_row] = A[max_p_row], A[col]
        P[col], P[max_p_row] = P[max_p_row], P[col]

        if max_p_row != col:
            s = s + 1

    return Tuple(A^, P^, s)


fn qr[
    dtype: DType
](owned A: Matrix[dtype]) raises -> Tuple[Matrix[dtype], Matrix[dtype]]:
    """
    Compute the QR decomposition of a matrix.

    Decompose the matrix `A` as `QR`, where `Q` is orthonormal and `R` is upper-triangular.
    This function is similar to `numpy.linalg.qr`.

    Args:
        A: The input matrix to be factorized.

    Returns:
        A tuple containing the orthonormal matrix `Q` and
        the upper-triangular matrix `R`.
    """
    var m = A.shape[0]
    var n = A.shape[1]

    var Q = Matrix.full[dtype](shape=(m, m))
    for i in range(m):
        Q._store(i, i, 1.0)

    var min_n = min(m, n)

    var H = Matrix.full[dtype](shape=(m, min_n))

    for i in range(min_n):
        compute_householder(H, A, i, i)
        compute_qr(H, i, A, i, i + 1)

    for i in range(min_n - 1, -1, -1):
        compute_qr(H, i, Q, i, i)

    return Q, A
