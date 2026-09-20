# ===----------------------------------------------------------------------=== #
# NuMojo: Solving
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""
Solving (numojo.routines.linalg.solving).
=========================================
Linear equation solvers.

Solver for `Ax = y` systems using LU decomposition algorithm, and matrix
inverse computation.

Exports
-------
- `solve`: Solve linear system.
- `inv`: Matrix inverse.
- `lstsq`: Least-squares solution to a linear system.
"""

# TODO: Add partial pivot support.
# TODO: Add determinant computation.

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from max.algorithm import parallelize

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import (
    eye,
    full,
    zeros,
)
from numojo.routines.linalg.decompositions import lu_decomposition, qr
from numojo.routines.linalg.products import matmul
from numojo.routines.manipulation import transpose


def forward_substitution[
    dtype: DType
](L: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """Perform forward substitution to solve `Lx = y`.

    Paramters:
        dtype: dtype of the resulting vector.

    Args:
        L: A lower triangular matrix.
        y: A vector.

    Returns:
        Solution to `Lx = y`. It is a vector.

    """

    # length of L
    var m = L.shape[0]

    # Initialize x
    var x = full[dtype](Shape(m), fill_value=SIMD[dtype, 1](0))

    for i in range(m):
        var value_on_hold: Scalar[dtype] = y.item(i)
        for j in range(i):
            value_on_hold = value_on_hold - L.item(i, j) * x.item(j)
        value_on_hold = value_on_hold / L.item(i, i)

        x.store(i, value_on_hold)

    return x^


def back_substitution[
    dtype: DType
](U: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]:
    """Perform forward substitution to solve `Ux = y`.

    Paramters:
        dtype: dtype of the resulting vector.

    Args:
        U: A upper triangular matrix.
        y: A vector.

    Returns:
        Solution to `Ux = y`. It is a vector.

    """

    # length of U
    var m = U.shape[0]
    # Initialize x
    var x = full[dtype](Shape(m), fill_value=SIMD[dtype, 1](0))

    for i in range(m - 1, -1, -1):
        var value_on_hold: Scalar[dtype] = y.item(i)
        for j in range(i + 1, m):
            value_on_hold = value_on_hold - U.item(i, j) * x.item(j)
        value_on_hold = value_on_hold / U.item(i, i)
        x.store(i, value_on_hold)

    return x^


def inv[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Find the inverse of a non-singular, row-major matrix.

    It uses the function `solve()` to solve `AB = I` for B, where I is
    an identity matrix.

    The speed is faster than numpy for matrices smaller than 100x100,
    and is slower for larger matrices.

    Parameters:
        dtype: Data type of the inverse matrix.

    Args:
        A: Input matrix. It should be non-singular, square, and row-major.

    Returns:
        The reversed matrix of the original matrix.

    """

    var m = A.shape[0]
    var I = eye[dtype](m, m)

    return solve(A, I)


def inv_lu[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """Find the inverse of a non-singular, row-major matrix.

    Use LU decomposition algorithm.

    The speed is faster than numpy for matrices smaller than 100x100,
    and is slower for larger matrices.

    TODO: Fix the issues in parallelization.
    `AX = I` where `I` is an identity matrix.

    Parameters:
        dtype: Data type of the inverse matrix.

    Args:
        array: Input matrix. It should be non-singular, square, and row-major.

    Returns:
        The reversed matrix of the original matrix.

    """

    var U: NDArray[dtype]
    var L: NDArray[dtype]
    var L_U: Tuple[NDArray[dtype], NDArray[dtype]] = lu_decomposition[dtype](
        array
    )
    L = L_U[0].copy()
    U = L_U[1].copy()

    var m = array.shape[0]

    var Y = eye[dtype](m, m)
    var Z = zeros[dtype](Shape(m, m))
    var X = zeros[dtype](Shape(m, m))

    @parameter
    def calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y.unsafe_load[width=1](i * m + col)
            for j in range(i):  # col of L
                _temp = _temp - L.unsafe_load[width=1](
                    i * m + j
                ) * Z.unsafe_load[width=1](j * m + col)
            _temp = _temp / L.unsafe_load[width=1](i * m + i)
            Z.unsafe_store[width=1](i * m + col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z.unsafe_load[width=1](i * m + col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U.unsafe_load[width=1](
                    i * m + j
                ) * X.unsafe_load[width=1](j * m + col)
            _temp2 = _temp2 / U.unsafe_load[width=1](i * m + i)
            X.unsafe_store[width=1](i * m + col, _temp2)

    parallelize[calculate_X](m, m)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    # var _Y = Y^
    # var _L = L^
    # var _U = U^

    return X^


def solve[
    dtype: DType
](A: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]:
    """Solve the linear system `AX = Y` for `X`.

    `A` should be a non-singular, row-major matrix (m x m).
    `Y` should be a matrix of (m x n).
    `X` is a matrix of (m x n).
    LU decomposition algorithm is adopted.

    The speed is faster than numpy for matrices smaller than 100x100,
    and is slower for larger matrices.

    For efficiency, `dtype` of the output array will be the same as the input
    arrays. Thus, use `astype()` before passing the arrays to this function.

    TODO: Use LAPACK for large matrices when it is available.

    Parameters:
        dtype: Data type of the inversed matrix.

    Args:
        A: Non-singular, square, and row-major matrix. The size is m x m.
        Y: Array of size m x n.

    Returns:
        Array of size m x n.

    An example goes as follows.

    ```mojo
    import numojo as nm
    def main() raises:
        var A = nm.fromstring("[[1, 0, 1], [0, 2, 1], [1, 1, 1]]")
        var B = nm.fromstring("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]")
        var X = nm.linalg.solve(A, B)
        print(X)
    ```
    ```console
    [[      -1.0    -1.0    2.0     ]
     [      -1.0    0.0     1.0     ]
     [      2.0     1.0     -2.0    ]]
    2-D array  Shape: [3, 3]  DType: float64
    ```

    The example is also a way to calculate inverse of matrix.

    """

    if not A.is_c_contiguous():
        return solve(A.contiguous(), Y)
    if not Y.is_c_contiguous():
        return solve(A, Y.contiguous())

    var U: NDArray[dtype]
    var L: NDArray[dtype]
    var L_U: Tuple[NDArray[dtype], NDArray[dtype]] = lu_decomposition[dtype](A)
    L = L_U[0].copy()
    U = L_U[1].copy()

    var m = A.shape[0]
    var n = Y.shape[1]

    var Z = zeros[dtype](Shape(m, n))
    var X = zeros[dtype](Shape(m, n))

    ####################################################################
    # Parallelization
    #
    # Parallelization does not work well since MAX 24.5.
    # This is because the ASAP destruction policy.
    # We temporarily use the variables to prolong their lifetime
    # TODO: Remove manual prolonging of lifetime in future if allowed.
    ####################################################################

    @parameter
    def calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y.unsafe_load[width=1](i * n + col)
            for j in range(i):  # col of L
                _temp = _temp - L.unsafe_load[width=1](
                    i * m + j
                ) * Z.unsafe_load[width=1](j * n + col)
            _temp = _temp / L.unsafe_load[width=1](i * m + i)
            Z.unsafe_store[width=1](i * n + col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z.unsafe_load[width=1](i * n + col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U.unsafe_load[width=1](
                    i * m + j
                ) * X.unsafe_load[width=1](j * n + col)
            _temp2 = _temp2 / U.unsafe_load[width=1](i * m + i)
            X.unsafe_store[width=1](i * n + col, _temp2)

    parallelize[calculate_X](n, n)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _L = L^
    var _U = U^
    var _Z = Z^
    var _m = m
    var _n = n

    return X^

    ####################################################################
    # Non-parallelization
    #
    # This approach does not adopt parallelization.
    ####################################################################

    # for col in range(n):
    #     # Solve `LZ = Y` for `Z` for each col
    #     for i in range(m):  # row of L
    #         var _temp = Y._buf.ptr.load(i * n + col)
    #         for j in range(i):  # col of L
    #             _temp = _temp - L._buf.ptr.load(i * m + j) * Z._buf.ptr.load(j * n + col)
    #         _temp = _temp / L._buf.ptr.load(i * m + i)
    #         Z._buf.ptr.store(i * n + col, _temp)

    #     # Solve `UZ = Z` for `X` for each col
    #     for i in range(m - 1, -1, -1):
    #         var _temp2 = Z._buf.ptr.load(i * n + col)
    #         for j in range(i + 1, m):
    #             _temp2 = _temp2 - U._buf.ptr.load(i * m + j) * X._buf.ptr.load(j * n + col)
    #         _temp2 = _temp2 / U._buf.ptr.load(i * m + i)
    #         X._buf.ptr.store(i * n + col, _temp2)

    # return X


def lstsq[
    dtype: DType
](A: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]:
    """Compute the least-squares solution to `Ax = Y`.

    `A` is an `(m, n)` matrix and `Y` is a vector of length `m` or a
    matrix of shape `(m, p)`. Returns `x` (shape `(n,)` or `(n, p)`,
    matching the shape of `Y`) minimizing `||Ax - Y||`.

    For overdetermined or square systems (`m >= n`) with `A` of full
    column rank, this uses QR decomposition. For underdetermined systems
    (`m < n`) with `A` of full row rank, this returns the minimum-norm
    solution via `A.T @ solve(A @ A.T, Y)`.

    This does not currently support rank-deficient `A`; for that case,
    use a SVD-based pseudoinverse instead once available.

    Parameters:
        dtype: Data type of the solution.

    Args:
        A: Coefficient matrix of shape `(m, n)`.
        Y: Right-hand side, of shape `(m,)` or `(m, p)`.

    Returns:
        The least-squares solution `x`, of shape `(n,)` or `(n, p)`.

    Examples:
        ```mojo
        import numojo as nm
        def main() raises:
            var A = nm.fromstring("[[1, 1], [1, 2], [1, 3]]")
            var y = nm.fromstring("[6, 0, 0]")
            var x = nm.linalg.lstsq(A, y)
            print(x)
        ```
    """

    if not A.is_c_contiguous():
        return lstsq(A.contiguous(), Y)
    if not Y.is_c_contiguous():
        return lstsq(A, Y.contiguous())

    var m = A.shape[0]
    var n = A.shape[1]

    var y_was_1d = Y.ndim == 1
    var Y2: NDArray[dtype] = Y.reshape(Shape(m, 1)) if y_was_1d else Y.copy()
    var p = Y2.shape[1]

    var X2: NDArray[dtype]
    if m >= n:
        var Q_R: Tuple[NDArray[dtype], NDArray[dtype]] = qr[dtype](A)
        var Q: NDArray[dtype] = Q_R[0].copy()
        var R: NDArray[dtype] = Q_R[1].copy()
        var Qty = matmul(transpose(Q), Y2)

        X2 = zeros[dtype](Shape(n, p))
        for c in range(p):
            var col_y = full[dtype](Shape(n), fill_value=SIMD[dtype, 1](0))
            for i in range(n):
                col_y.store(i, Qty.item(i, c))
            var x_col = back_substitution(R, col_y)
            for i in range(n):
                X2.store(i, c, val=x_col.item(i))
    else:
        var A_T = transpose(A)
        var AAT = matmul(A, A_T)
        var Z = solve(AAT, Y2)
        X2 = matmul(A_T, Z)

    if y_was_1d:
        return X2.reshape(Shape(n))
    return X2^


# TODO: remove unnecessary copies going on here later.
