from .mat import *
from .creation import *

# ===-----------------------------------------------------------------------===#
# Fucntions for linear algebra
# ===-----------------------------------------------------------------------===#


fn lu_decomposition[
    dtype: DType
](A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype]]:
    # Check whether the matrix is square
    try:
        if A.shape[0] != A.shape[1]:
            raise Error("The matrix is not square!")
    except e:
        print(e)
    var n = A.shape[0]

    # Initiate upper and lower triangular matrices
    var U = full[dtype](shape=(n, n))
    var L = full[dtype](shape=(n, n))

    # Fill in L and U
    for i in range(0, n):
        for j in range(i, n):
            # Fill in L
            if i == j:
                L[i, i] = 1
            else:
                var sum_of_products_for_L: Scalar[dtype] = 0
                for k in range(0, i):
                    sum_of_products_for_L += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - sum_of_products_for_L) / U[i, i]

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum_of_products_for_U

    return L, U


fn solve[dtype: DType](A: Matrix[dtype], Y: Matrix[dtype]) -> Matrix[dtype]:
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
            var _temp = Y[i, col]
            for j in range(i):  # col of L
                _temp = _temp - L[i, j] * Z[j, col]
            _temp = _temp / L[i, i]
            Z[i, col] = _temp

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z[i, col]
            for j in range(i + 1, m):
                _temp2 = _temp2 - U[i, j] * X[j, col]
            _temp2 = _temp2 / U[i, i]
            X[i, col] = _temp2

    parallelize[calculate_X](n, n)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _L = L^
    var _U = U^
    var _Z = Z^
    var _m = m
    var _n = n

    return X^


fn inv[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Inverse of matrix."""

    # Check whether the matrix is square
    try:
        if A.shape[0] != A.shape[1]:
            raise Error("The matrix is not square!")
    except e:
        print(e)

    var I = identity[dtype](A.shape[0])
    var B = solve(A, I)

    return B^


fn transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Transpose of a matrix."""

    var B = Matrix[dtype](Tuple(A.shape[1], A.shape[0]))

    if A.shape[0] == 1 or A.shape[1] == 1:
        memcpy(B._buf, A._buf, A.size)
    else:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B._store(i, j, A._load(j, i))
    return B^


fn matmul[dtype: DType](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]:
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

    try:
        if A.shape[1] != B.shape[0]:
            raise Error("The shapes of matrices do not match!")
    except e:
        print(e)
        print("`matmul` error.")

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


fn lstsq[dtype: DType](X: Matrix[dtype], y: Matrix[dtype]) -> Matrix[dtype]:
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

    try:
        if X.shape[0] != y.shape[0]:
            var message = "The row number of `X` {X.shape[0]} does not match the row number of `y` {y.shape[0]}"
            raise Error(message)
    except e:
        print(e)
    var X_prime = X.T()
    var b = (X_prime @ X).inv() @ X_prime @ y
    return b^
