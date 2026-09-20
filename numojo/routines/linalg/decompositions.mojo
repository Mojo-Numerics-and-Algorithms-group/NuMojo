# ===----------------------------------------------------------------------=== #
# NuMojo: Decompositions
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Decompositions (numojo.routines.linalg.decompositions).
=======================================================
Matrix decomposition operations.

Functions for decomposing NDArrays representing 2-D matrices (LU
decomposition with partial pivoting).

Exports
-------
- `lu_decomposition`: LU factorization.
- `qr`: QR factorization (Householder reflections).
- `cholesky`: Cholesky factorization.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import (
    full,
    identity,
)


def lu_decomposition[
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
    def main() raises:
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
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location="lu_decomposition",
            )
        )

    # Check whether the matrix is square
    var shape_of_array: NDArrayShape = A.shape
    if shape_of_array[0] != shape_of_array[1]:
        raise Error(
            NumojoError(
                category="shape",
                message="The matrix is not square!",
                location="lu_decomposition",
            )
        )
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
    # def calculate(i: Int):
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


def partial_pivoting[
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
        raise Error(
            NumojoError(
                category="shape",
                message=String("Array must be 2d."),
                location="partial_pivoting",
            )
        )
    if A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message=String("Array is not square."),
                location="partial_pivoting",
            )
        )

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


def qr[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """
    Perform QR decomposition of a matrix using Householder reflections.

    Computes the "reduced" (economy) factorization, matching
    `numpy.linalg.qr` with `mode="reduced"` (the default): for an input
    of shape `(m, n)`, `Q` has shape `(m, k)` with orthonormal columns
    and `R` has shape `(k, n)` and is upper triangular, where
    `k = min(m, n)`.

    Parameters:
        dtype: Data type of the input and output matrices. Should be a
            floating-point type.

    Args:
        A: Input matrix of shape `(m, n)`.

    Returns:
        A tuple `(Q, R)` such that `Q @ R` reconstructs `A` and
        `Q.T @ Q` is the identity matrix.

    Raises:
        NumojoError: If the array is not 2-dimensional.

    Examples:
        ```mojo
        import numojo as nm
        def main() raises:
            var A = nm.fromstring("[[1, 2], [3, 4], [5, 6]]")
            var Q: nm.NDArray
            var R: nm.NDArray
            Q, R = nm.linalg.qr(A)
            print(Q @ R)  # reconstructs A
        ```
    """

    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location="qr",
            )
        )

    var m = A.shape[0]
    var n = A.shape[1]
    var k = min(m, n)

    var R = A.copy() if A.is_c_contiguous() else A.contiguous()
    var Q = identity[dtype](m)

    for col in range(k):
        var h = m - col  # length of the active Householder vector

        var norm_x: Scalar[dtype] = 0
        for i in range(col, m):
            var val = R.item(i, col)
            norm_x += val * val
        norm_x = sqrt(norm_x)
        if norm_x == 0:
            continue

        var x0 = R.item(col, col)
        var alpha: Scalar[dtype] = -norm_x if x0 >= 0 else norm_x

        var v = List[Scalar[dtype]](capacity=h)
        for i in range(col, m):
            v.append(R.item(i, col))
        v[0] = v[0] - alpha

        var v_norm: Scalar[dtype] = 0
        for i in range(h):
            v_norm += v[i] * v[i]
        v_norm = sqrt(v_norm)
        if v_norm == 0:
            continue
        for i in range(h):
            v[i] = v[i] / v_norm

        # R[col:, :] -= 2 * v * (v^T @ R[col:, :])
        for j in range(n):
            var dot_val: Scalar[dtype] = 0
            for i in range(h):
                dot_val += v[i] * R.item(col + i, j)
            for i in range(h):
                R[Item(col + i, j)] = R.item(col + i, j) - 2 * v[i] * dot_val

        # Q[:, col:] -= 2 * (Q[:, col:] @ v) * v^T
        for i in range(m):
            var dot_val2: Scalar[dtype] = 0
            for j in range(h):
                dot_val2 += Q.item(i, col + j) * v[j]
            for j in range(h):
                Q[Item(i, col + j)] = Q.item(i, col + j) - 2 * dot_val2 * v[j]

    var Q_reduced = NDArray[dtype](Shape(m, k))
    for i in range(m):
        for j in range(k):
            Q_reduced[Item(i, j)] = Q.item(i, j)

    var R_reduced = NDArray[dtype](Shape(k, n))
    for i in range(k):
        for j in range(n):
            R_reduced[Item(i, j)] = R.item(i, j) if j >= i else 0

    return Q_reduced^, R_reduced^


def cholesky[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the Cholesky decomposition of a symmetric positive-definite
    matrix.

    Finds the lower triangular matrix `L` such that `A = L @ L.T`,
    matching `numpy.linalg.cholesky`.

    Parameters:
        dtype: Data type of the input and output matrices. Should be a
            floating-point type.

    Args:
        A: Symmetric positive-definite matrix of shape `(n, n)`.

    Returns:
        The lower triangular Cholesky factor `L` of shape `(n, n)`.

    Raises:
        NumojoError: If the array is not 2-dimensional or not square.
        NumojoError: If the array is not positive-definite (a
            non-positive value is encountered on the diagonal during
            factorization).

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var A = nm.fromstring("[[4, 2], [2, 3]]")
        var L = nm.linalg.cholesky(A)
        print(L @ L.T)  # reconstructs A
    ```
    """

    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="The array is not 2-dimensional!",
                location="cholesky",
            )
        )
    if A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message="The matrix is not square!",
                location="cholesky",
            )
        )

    var n = A.shape[0]
    var L = full[dtype](shape=Shape(n, n), fill_value=SIMD[dtype, 1](0))

    for i in range(n):
        for j in range(i + 1):
            var sum_of_products: Scalar[dtype] = 0
            for kk in range(j):
                sum_of_products += L.item(i, kk) * L.item(j, kk)

            if i == j:
                var diag_val = A.item(i, i) - sum_of_products
                if diag_val <= 0:
                    raise Error(
                        NumojoError(
                            category="value",
                            message="The matrix is not positive-definite.",
                            location="cholesky",
                        )
                    )
                L[Item(i, j)] = sqrt(diag_val)
            else:
                L[Item(i, j)] = (A.item(i, j) - sum_of_products) / L.item(j, j)

    return L^
