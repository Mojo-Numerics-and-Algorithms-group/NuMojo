"""
Linear Algebra Solver

Provides:
    - Solver of `Ax = y` using LU decomposition algorithm.
    - Inverse of an invertible matrix.

# TODO:
    - Partial pivot.
    - Determinant.
"""

from numojo.core.ndarray import NDArray
from numojo.core.index import Idx
from numojo.routines.creation import zeros, eye, full
from algorithm import parallelize


fn forward_substitution[
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

    return x


fn back_substitution[
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

    return x


fn inv[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """Find the inverse of a non-singular, row-major matrix.

    It uses the function `solve()` to solve `AB = I` for B, where I is
    an identity matrix.

    The speed is faster than numpy for matrices smaller than 100x100,
    and is slower for larger matrices.

    Parameters:
        dtype: Data type of the inversed matrix.

    Args:
        A: Input matrix. It should be non-singular, square, and row-major.

    Returns:
        The reversed matrix of the original matrix.

    """

    var m = A.shape[0]
    var I = eye[dtype](m, m)

    return solve(A, I)


fn inv_raw[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """Find the inverse of a non-singular, square matrix.

    WARNING: This function is slower than `inv`.
    as it does not adopt parallelization by using raw methods.

    Parameters:
        dtype: Data type of the inversed matrix.

    Args:
        array: Input matrix. It should be non-singular and square.

    Returns:
        The reversed matrix of the original matrix.

    An example goes as follows:

    ```
    import numojo as nm
    fn main() raises:
        var A = nm.NDArray("[[1,0,1], [0,2,1], [1,1,1]]")
        var B = nm.math.linalg.solver.inv_raw(A)
        print("Original matrix:")
        print(A)
        print("Reversed matrix:")
        print(B)
        print("Verify whether AB = I:")
        print(A @ B)
    ```
    ```console
    Original matrix:
    [[      1.0     0.0     1.0     ]
     [      0.0     2.0     1.0     ]
     [      1.0     1.0     1.0     ]]
    2-D array  Shape: [3, 3]  DType: float64
    Reversed matrix:
    [[      -1.0    -1.0    2.0     ]
     [      -1.0    0.0     1.0     ]
     [      2.0     1.0     -2.0    ]]
    2-D array  Shape: [3, 3]  DType: float64
    Verify whether AB = I:
    [[      1.0     0.0     0.0     ]
     [      0.0     1.0     0.0     ]
     [      0.0     0.0     1.0     ]]
    2-D array  Shape: [3, 3]  DType: float64
    ```

    TODO: Optimize the speed.

    """

    var U: NDArray[dtype]
    var L: NDArray[dtype]
    L, U = lu_decomposition[dtype](array)

    var m = array.shape[0]
    var inversed = NDArray[dtype](shape=array.shape)

    # Initialize vectors
    var y = NDArray[dtype](Shape(m))
    var z = NDArray[dtype](Shape(m))
    var x = NDArray[dtype](Shape(m))

    for i in range(m):
        # Each time, one of the item is changed to 1
        y = zeros[dtype](Shape(m))
        y.store(i, Scalar[dtype](1))

        # Solve `Lz = y` for `z`
        z = forward_substitution(L, y)

        # Solve `Ux = z` for `x`
        x = back_substitution(U, z)

        # print("z2", z)
        # print("x2", x)

        for j in range(m):
            inversed.__setitem__(Idx(j, i), x.item(j))

    return inversed


fn inv_lu[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """Find the inverse of a non-singular, row-major matrix.

    Use LU decomposition algorithm.

    The speed is faster than numpy for matrices smaller than 100x100,
    and is slower for larger matrices.

    TODO: Fix the issues in parallelization.
    `AX = I` where `I` is an identity matrix.

    Parameters:
        dtype: Data type of the inversed matrix.

    Args:
        array: Input matrix. It should be non-singular, square, and row-major.

    Returns:
        The reversed matrix of the original matrix.

    """

    var U: NDArray[dtype]
    var L: NDArray[dtype]
    L, U = lu_decomposition[dtype](array)

    var m = array.shape[0]

    var Y = eye[dtype](m, m)
    var Z = zeros[dtype](Shape(m, m))
    var X = zeros[dtype](Shape(m, m))

    @parameter
    fn calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y._buf.ptr.load(i * m + col)
            for j in range(i):  # col of L
                _temp = _temp - L._buf.ptr.load(i * m + j) * Z._buf.ptr.load(
                    j * m + col
                )
            _temp = _temp / L._buf.ptr.load(i * m + i)
            Z._buf.ptr.store(i * m + col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z._buf.ptr.load(i * m + col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U._buf.ptr.load(i * m + j) * X._buf.ptr.load(
                    j * m + col
                )
            _temp2 = _temp2 / U._buf.ptr.load(i * m + i)
            X._buf.ptr.store(i * m + col, _temp2)

    parallelize[calculate_X](m, m)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _Y = Y^
    var _L = L^
    var _U = U^

    return X


fn solve[
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
        Y: Matrix of size m x n.

    Returns:
        Matrix of size m x n.

    An example goes as follows.

    ```mojo
    import numojo as nm
    fn main() raises:
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

    var U: NDArray[dtype]
    var L: NDArray[dtype]
    L, U = lu_decomposition[dtype](A)

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
    fn calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y._buf.ptr.load(i * n + col)
            for j in range(i):  # col of L
                _temp = _temp - L._buf.ptr.load(i * m + j) * Z._buf.ptr.load(
                    j * n + col
                )
            _temp = _temp / L._buf.ptr.load(i * m + i)
            Z._buf.ptr.store(i * n + col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z._buf.ptr.load(i * n + col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U._buf.ptr.load(i * m + j) * X._buf.ptr.load(
                    j * n + col
                )
            _temp2 = _temp2 / U._buf.ptr.load(i * m + i)
            X._buf.ptr.store(i * n + col, _temp2)

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
