# ===----------------------------------------------------------------------=== #
# Decompositions
# ===----------------------------------------------------------------------=== #
from sys import simd_width_of
from algorithm import parallelize, vectorize
from memory import UnsafePointer, memcpy, memset_zero

import math as builtin_math

from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix, issymmetric, MatrixImpl
from numojo.routines.creation import zeros, eye, full


@always_inline
fn _compute_householder[
    dtype: DType
](mut H: Matrix[dtype], mut R: Matrix[dtype], work_index: Int) raises -> None:
    alias simd_width = simd_width_of[dtype]()
    alias sqrt2: Scalar[dtype] = 1.4142135623730951
    var rRows = R.shape[0]

    @parameter
    fn load_store_vec[n_elements: Int](i: Int):
        var r_value = R._load[n_elements](i + work_index, work_index)
        H._store[n_elements](i + work_index, work_index, r_value)
        R._store[n_elements](i + work_index, work_index, 0.0)

    vectorize[load_store_vec, simd_width](rRows - work_index)

    var norm = Scalar[dtype](0)

    @parameter
    fn calculate_norm[width: Int](i: Int):
        norm += (H._load[width=width](i, work_index) ** 2).reduce_add()

    vectorize[calculate_norm, simd_width](rRows)

    norm = builtin_math.sqrt(norm)
    if work_index == rRows - 1 or norm == 0:
        first_element = H._load(work_index, work_index)
        R._store(work_index, work_index, -first_element)
        H._store(work_index, work_index, sqrt2)
        return

    var scaling_factor = 1.0 / norm
    if H._load(work_index, work_index) < 0:
        scaling_factor = -scaling_factor

    R._store(work_index, work_index, -1 / scaling_factor)

    @parameter
    fn scaling_factor_vec[simd_width: Int](i: Int):
        H._store[simd_width](
            i, work_index, H._load[simd_width](i, work_index) * scaling_factor
        )

    vectorize[scaling_factor_vec, simd_width](rRows)

    increment = H._load(work_index, work_index) + 1.0
    H._store(work_index, work_index, increment)

    scaling_factor = builtin_math.sqrt(1.0 / increment)

    @parameter
    fn scaling_factor_increment_vec[simd_width: Int](i: Int):
        H._store[simd_width](
            i, work_index, H._load[simd_width](i, work_index) * scaling_factor
        )

    vectorize[scaling_factor_increment_vec, simd_width](rRows)


@always_inline
fn _apply_householder[
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
    alias simdwidth = simd_width_of[dtype]()
    for j in range(column_start, aCols):
        var dot: SIMD[dtype, 1] = 0.0

        @parameter
        fn calculate_norm[width: Int](i: Int):
            dot += (
                H._load[width=width](i, work_index) * A._load[width=width](i, j)
            ).reduce_add()

        vectorize[calculate_norm, simdwidth](aRows)

        @parameter
        fn closure[width: Int](i: Int):
            val = A._load[width](i, j) - H._load[width](i, work_index) * dot
            A._store(i, j, val)

        vectorize[closure, simdwidth](aRows)


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
    var shape_of_array: NDArrayShape = A.shape
    if shape_of_array[0] != shape_of_array[1]:
        raise ("The matrix is not square!")
    var n: Int = shape_of_array[0]

    # Check whether the matrix is singular
    # if singular:
    #     raise("The matrix is singular!")

    # Change dtype of array to defined dtype
    # var A = array.astype[dtype]()

    # Initiate upper and lower triangular matrices
    var U: NDArray[dtype] = full[dtype](
        shape=shape_of_array, fill_value=SIMD[dtype, 1](0)
    )
    var L: NDArray[dtype] = full[dtype](
        shape=shape_of_array, fill_value=SIMD[dtype, 1](0)
    )

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

    return L^, U^


fn lu_decomposition[
    dtype: DType
](A: MatrixImpl[dtype, **_]) raises -> Tuple[Matrix[dtype], Matrix[dtype]]:
    """
    Perform LU (lower-upper) decomposition for matrix.
    """
    # Check whether the matrix is square
    if A.shape[0] != A.shape[1]:
        raise Error(
            String("{}x{} matrix is not square.").format(A.shape[0], A.shape[1])
        )

    var n: Int = A.shape[0]

    # Initiate upper and lower triangular matrices
    var U: Matrix[dtype] = Matrix.zeros[dtype](shape=(n, n), order=A.order())
    var L: Matrix[dtype] = Matrix.zeros[dtype](shape=(n, n), order=A.order())

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

    return L^, U^


fn partial_pivoting[
    dtype: DType
](var A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype], Int]:
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
](A: MatrixImpl[dtype, **_]) raises -> Tuple[Matrix[dtype], Matrix[dtype], Int]:
    """
    Perform partial pivoting for matrix.
    """
    var n = A.shape[0]
    # Work on a copy that preserves the original layout
    var result = A.create_copy()
    var P = Matrix.identity[dtype](n, order=A.order())
    var s: Int = 0  # Number of row exchanges

    for col in range(n):
        var max_p = abs(result[col, col])
        var max_p_row = col
        for row in range(col + 1, n):
            if abs(result[row, col]) > max_p:
                max_p = abs(result[row, col])
                max_p_row = row

        if max_p_row != col:
            # Swap rows in result and permutation matrix using element-wise swap
            for j in range(n):
                var t = result._load(col, j)
                result._store(col, j, result._load(max_p_row, j))
                result._store(max_p_row, j, t)
                var tp = P._load(col, j)
                P._store(col, j, P._load(max_p_row, j))
                P._store(max_p_row, j, tp)
            s = s + 1

    return Tuple(result^, P^, s)


fn qr[
    dtype: DType
](A: Matrix[dtype], mode: String = "reduced") raises -> Tuple[
    Matrix[dtype], Matrix[dtype]
]:
    """
    Computes the QR decomposition using Householder transformations. For best
    performance, the input matrix should be in column-major order.

    Args:
        A: The input matrix.
        mode: The mode of the decomposition. Can be "complete" or "reduced" simillar to numpy's QR decomposition.
            - "complete": Returns Q and R such that A = QR, where Q is m x m and R is m x n.
            - "reduced": Returns Q and R such that A = QR, where Q is m x min(m,n) and R is min(m,n) x n.
    Returns:
        A tuple `(Q, R)` where `Q` is orthonormal and `R` is upper-triangular.
    """
    var inner: Int
    var reorder: Bool = False
    var reduce: Bool

    var m = A.shape[0]
    var n = A.shape[1]

    var min_n = min(m, n)

    if mode == "complete" or m == n:
        reduce = False
        inner = m
    elif mode == "reduced":
        reduce = True
        inner = min_n
    else:
        raise Error(String("Invalid mode: {}").format(mode))

    var R: Matrix[dtype, OwnData]

    if A.flags.C_CONTIGUOUS:
        reorder = True

    if reorder:
        R = A.reorder_layout()
    else:
        R = Matrix.zeros[dtype](shape=(m, n), order="F")
        for i in range(m):
            for j in range(n):
                R._store(i, j, A._load(i, j))

    var H = Matrix.zeros[dtype](shape=(m, min_n), order="F")

    for i in range(min_n):
        _compute_householder(H, R, i)
        _apply_householder(H, i, R, i, i + 1)

    var Q = Matrix.zeros[dtype]((m, inner), order="F")
    for i in range(inner):
        Q[i, i] = 1.0

    for i in range(min_n - 1, -1, -1):
        _apply_householder(H, i, Q, i, i)

    if reorder:
        var Q_reordered = Q.reorder_layout()
        if reduce:
            var R_reduced = Matrix.zeros[dtype](shape=(inner, n), order="C")
            for i in range(inner):
                for j in range(n):
                    R_reduced._store(i, j, R._load(i, j))
            return Q_reordered^, R_reduced^
        else:
            var R_reordered = R.reorder_layout()
            return Q_reordered^, R_reordered^
    else:
        if reduce:
            var R_reduced = Matrix.zeros[dtype](shape=(inner, n), order="F")
            for i in range(inner):
                for j in range(n):
                    R_reduced._store(i, j, R._load(i, j))
            return Q^, R_reduced^
        else:
            return Q^, R^


# ===----------------------------------------------------------------------=== #
# Eigenvalue Decomposition (symmetric) via the QR algorithm
# ===----------------------------------------------------------------------=== #


fn eig[
    dtype: DType
](
    A: Matrix[dtype],
    tol: Scalar[dtype] = 1.0e-12,
    max_iter: Int = 10000,
) raises -> Tuple[Matrix[dtype], Matrix[dtype]]:
    """
    Computes the eigenvalue decomposition for symmetric matrices using the QR algorithm.
    For best performance, the input matrix should be in column-major order.

    Args:
        A: The input matrix. Must be square and symmetric.
        tol: Convergence tolerance for off-diagonal elements.
        max_iter: Maximum number of iterations for the QR algorithm.

    Returns:
        A tuple `(Q, D)` where:
            - Q: A matrix whose columns are the eigenvectors
            - D: A diagonal matrix containing the eigenvalues

    Raises:
        Error: If the matrix is not square or symmetric.
        Error: If the algorithm does not converge within max_iter iterations.
    """
    if A.shape[0] != A.shape[1]:
        raise Error("Matrix is not square.")

    var n = A.shape[0]
    if not issymmetric(A):
        raise Error("Matrix is not symmetric.")

    var T: Matrix[dtype]
    if A.flags.C_CONTIGUOUS:
        T = A.reorder_layout()
    else:
        T = A.copy()

    var Q_total = Matrix.identity[dtype](n)

    for _k in range(max_iter):
        var Qk: Matrix[dtype]
        var Rk: Matrix[dtype]
        var matrices: Tuple[Matrix[dtype], Matrix[dtype]] = qr(
            T, mode="complete"
        )
        Qk = matrices[0].copy()
        Rk = matrices[1].copy()

        T = Rk @ Qk
        Q_total = Q_total @ Qk

        var offdiag_norm: Scalar[dtype] = 0
        for i in range(n):
            for j in range(i + 1, n):
                var v = T._load(i, j)
                offdiag_norm += v * v
        if builtin_math.sqrt(offdiag_norm) < tol:
            break
    else:
        raise Error(
            String("QR algorithm did not converge in {} iterations.").format(
                max_iter
            )
        )

    var D = Matrix.zeros[dtype](shape=(n, n), order=A.order())
    for i in range(n):
        D._store(i, i, T._load(i, i))

    if A.flags.C_CONTIGUOUS:
        Q_total = Q_total.reorder_layout()

    return Q_total^, D^
