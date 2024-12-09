"""
`numojo.mat.linalg` module provides functions for linear algebra.

- Matrix and vector products
- Decompositions
- Matrix eigenvalues
- Norms and other numbers
- Solving equations and inverting matrices
- Matrix operations

"""

from .matrix import *
from .creation import *

# ===----------------------------------------------------------------------===#
# Matrix and vector products
# ===----------------------------------------------------------------------===#


fn matmul[
    dtype: DType
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[dtype]:
    """Matrix multiplication.

    See `numojo.math.linalg.matmul.matmul_parallelized()`.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(1000, 1000))
    var B = mat.rand(shape=(1000, 1000))
    var C = mat.matmul(A, B)
    ```
    """

    alias width = max(simdwidthof[dtype](), 16)

    if A.shape[1] != B.shape[0]:
        raise Error(
            String("Cannot matmul {}x{} matrix with {}x{} matrix.").format(
                A.shape[0], A.shape[1], B.shape[0], B.shape[1]
            )
        )

    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]
    var C: Matrix[dtype] = zeros[dtype](shape=(t0, t2))

    @parameter
    fn calculate_CC(m: Int):
        for k in range(t1):

            @parameter
            fn dot[simd_width: Int](n: Int):
                C._store[simd_width](
                    m,
                    n,
                    C._load[simd_width](m, n)
                    + A._load(m, k) * B._load[simd_width](k, n),
                )

            vectorize[dot, width](t2)

    parallelize[calculate_CC](t0, t0)

    var _t0 = t0
    var _t1 = t1
    var _t2 = t2
    var _A = A
    var _B = B

    return C^


# ===----------------------------------------------------------------------===#
# Decompositions
# ===----------------------------------------------------------------------===#


fn partial_pivoting[
    dtype: DType
](owned A: Matrix[dtype]) raises -> Tuple[Matrix[dtype], Matrix[dtype], Int]:
    """Perform partial pivoting."""
    var n = A.shape[0]
    var P = identity[dtype](n)
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


fn lu_decomposition[
    dtype: DType
](A: Matrix[dtype]) raises -> Tuple[Matrix[dtype], Matrix[dtype]]:
    # Check whether the matrix is square
    if A.shape[0] != A.shape[1]:
        raise Error(
            String("{}x{} matrix is not square.").format(A.shape[0], A.shape[1])
        )

    var n = A.shape[0]

    # Initiate upper and lower triangular matrices
    var U = full[dtype](shape=(n, n))
    var L = full[dtype](shape=(n, n))

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


# ===----------------------------------------------------------------------===#
# Norms and other numbers
# ===----------------------------------------------------------------------===#


fn det[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find the determinant of A using LUP decomposition.
    """
    var det_L: Scalar[dtype] = 1
    var det_U: Scalar[dtype] = 1
    var n = A.shape[0]  # Dimension of the matrix

    var U: Matrix[dtype]
    var L: Matrix[dtype]
    A_pivoted, _, s = partial_pivoting(A)
    L, U = lu_decomposition[dtype](A_pivoted)

    for i in range(n):
        det_L = det_L * L[i, i]
        det_U = det_U * U[i, i]

    if s % 2 == 0:
        return det_L * det_U
    else:
        return -det_L * det_U


fn trace[
    dtype: DType
](A: Matrix[dtype], offset: Int = 0) raises -> Scalar[dtype]:
    """
    Return the sum along diagonals of the array.

    Similar to `numpy.trace`.
    """
    var m = A.shape[0]
    var n = A.shape[1]

    if offset >= max(m, n):  # Offset beyond the shape of the matrix
        return 0

    var res = Scalar[dtype](0)

    if offset >= 0:
        for i in range(n - offset):
            res = res + A[i, i + offset]
    else:
        for i in range(m + offset):
            res = res + A[i - offset, i]

    return res


# ===----------------------------------------------------------------------===#
# Solving equations and inverting matrices
# ===----------------------------------------------------------------------===#


fn inv[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Inverse of matrix.
    """

    # Check whether the matrix is square
    if A.shape[0] != A.shape[1]:
        raise Error(
            String("{}x{} matrix is not square.").format(A.shape[0], A.shape[1])
        )

    var I = identity[dtype](A.shape[0])
    var B = solve(A, I)

    return B^


fn lstsq[
    dtype: DType
](X: Matrix[dtype], y: Matrix[dtype]) raises -> Matrix[dtype]:
    """Caclulate the OLS estimates.

    Example:
    ```mojo
    from numojo import mat
    X = mat.rand((1000000, 5))
    y = mat.rand((1000000, 1))
    print(mat.lstsq(X, y))
    ```
    ```console
    [[0.18731374756029967]
     [0.18821352688798607]
     [0.18717162200411439]
     [0.1867570378683612]
     [0.18828715376701158]]
    Size: 5x1  DType: float64
    ```
    """

    if X.shape[0] != y.shape[0]:
        raise Error(
            String(
                "Row number of `X` {X.shape[0]} should equal that of `y`"
                " {y.shape[0]}"
            )
        )

    var X_prime = X.T()
    var b = (X_prime @ X).inv() @ X_prime @ y
    return b^


fn solve[
    dtype: DType
](A: Matrix[dtype], Y: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Solve `AX = Y` using LUP decomposition.
    """
    var U: Matrix[dtype]
    var L: Matrix[dtype]
    A_pivoted, P, _ = partial_pivoting(A)
    L, U = lu_decomposition[dtype](A_pivoted)

    var m = A.shape[0]
    var n = Y.shape[1]

    var Z = full[dtype]((m, n))
    var X = full[dtype]((m, n))

    var PY = P @ Y

    @parameter
    fn calculate_X(col: Int) -> None:
        # Solve `LZ = PY` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = PY._load(i, col)
            for j in range(i):  # col of L
                _temp = _temp - L._load(i, j) * Z._load(j, col)
            _temp = _temp / L._load(i, i)
            Z._store(i, col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z._load(i, col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U._load(i, j) * X._load(j, col)
            _temp2 = _temp2 / U._load(i, i)
            X._store(i, col, _temp2)

    parallelize[calculate_X](n, n)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _L = L^
    var _U = U^
    var _Z = Z^
    var _PY = PY^
    var _m = m
    var _n = n

    return X^


fn solve_lu[
    dtype: DType
](A: Matrix[dtype], Y: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Solve `AX = Y` using LU decomposition.
    """
    var U: Matrix[dtype]
    var L: Matrix[dtype]
    L, U = lu_decomposition[dtype](A)

    var m = A.shape[0]
    var n = Y.shape[1]

    var Z = full[dtype]((m, n))
    var X = full[dtype]((m, n))

    @parameter
    fn calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y._load(i, col)
            for j in range(i):  # col of L
                _temp = _temp - L._load(i, j) * Z._load(j, col)
            _temp = _temp / L._load(i, i)
            Z._store(i, col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z._load(i, col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U._load(i, j) * X._load(j, col)
            _temp2 = _temp2 / U._load(i, i)
            X._store(i, col, _temp2)

    parallelize[calculate_X](n, n)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _L = L^
    var _U = U^
    var _Z = Z^
    var _m = m
    var _n = n

    return X^


# ===----------------------------------------------------------------------===#
# Matrix operations
# ===----------------------------------------------------------------------===#


fn transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Transpose of matrix.
    """

    var B = Matrix[dtype](Tuple(A.shape[1], A.shape[0]))

    if A.shape[0] == 1 or A.shape[1] == 1:
        memcpy(B._buf, A._buf, A.size)
    else:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B._store(i, j, A._load(j, i))
    return B^
