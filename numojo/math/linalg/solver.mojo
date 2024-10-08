"""
Linear Algebra Solver

Provides:
    - Solver of `Ax = y` using LU decomposition algorithm.
    - Inverse of an invertible matrix.
"""

from ...core.ndarray import NDArray
from ...core.array_creation_routines import zeros, eye
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
    var shape_of_array = A.shape()
    if shape_of_array[0] != shape_of_array[1]:
        raise ("The matrix is not square!")
    var n = shape_of_array[0]

    # Check whether the matrix is singular
    # if singular:
    #     raise("The matrix is singular!")

    # Change dtype of array to defined dtype
    # var A = array.astype[dtype]()

    # Initiate upper and lower triangular matrices
    var U = NDArray[dtype](shape=shape_of_array, fill=SIMD[dtype, 1](0))
    var L = NDArray[dtype](shape=shape_of_array, fill=SIMD[dtype, 1](0))

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
    var m = L.shape()[0]

    # Initialize x
    var x = NDArray[dtype](m, fill=SIMD[dtype, 1](0))

    for i in range(m):
        var value_on_hold: Scalar[dtype] = y.item(i)
        for j in range(i):
            value_on_hold = value_on_hold - L.item(i, j) * x.item(j)
        value_on_hold = value_on_hold / L.item(i, i)

        x.__setitem__(i, value_on_hold)

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
    var m = U.shape()[0]
    # Initialize x
    var x = NDArray[dtype](m, fill=SIMD[dtype, 1](0))

    for i in range(m - 1, -1, -1):
        var value_on_hold: Scalar[dtype] = y.item(i)
        for j in range(i + 1, m):
            value_on_hold = value_on_hold - U.item(i, j) * x.item(j)
        value_on_hold = value_on_hold / U.item(i, i)
        x.__setitem__(i, value_on_hold)

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

    var m = A.shape()[0]
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

    var m = array.shape()[0]
    var inversed = NDArray[dtype](shape=array.shape())

    # Initialize vectors
    var y = NDArray[dtype](m)
    var z = NDArray[dtype](m)
    var x = NDArray[dtype](m)

    for i in range(m):
        # Each time, one of the item is changed to 1
        y = NDArray[dtype](m, fill=Scalar[dtype](0))
        y.__setitem__(i, Scalar[dtype](1))

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

    var m = array.shape()[0]

    var Y = eye[dtype](m, m)
    var Z = zeros[dtype](m, m)
    var X = zeros[dtype](m, m)

    @parameter
    fn calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y.load(i * m + col)
            for j in range(i):  # col of L
                _temp = _temp - L.load(i * m + j) * Z.load(j * m + col)
            _temp = _temp / L.load(i * m + i)
            Z.store(i * m + col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z.load(i * m + col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U.load(i * m + j) * X.load(j * m + col)
            _temp2 = _temp2 / U.load(i * m + i)
            X.store(i * m + col, _temp2)

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
        var A = nm.NDArray(str("[[1, 0, 1], [0, 2, 1], [1, 1, 1]]"))
        var B = nm.NDArray(str("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"))
        var X = nm.solver.solve(A, B)
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

    var m = A.shape()[0]
    var n = Y.shape()[1]

    var Z = zeros[dtype](m, n)
    var X = zeros[dtype](m, n)

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
            var _temp = Y.load(i * n + col)
            for j in range(i):  # col of L
                _temp = _temp - L.load(i * m + j) * Z.load(j * n + col)
            _temp = _temp / L.load(i * m + i)
            Z.store(i * n + col, _temp)

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z.load(i * n + col)
            for j in range(i + 1, m):
                _temp2 = _temp2 - U.load(i * m + j) * X.load(j * n + col)
            _temp2 = _temp2 / U.load(i * m + i)
            X.store(i * n + col, _temp2)

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
    #         var _temp = Y.load(i * n + col)
    #         for j in range(i):  # col of L
    #             _temp = _temp - L.load(i * m + j) * Z.load(j * n + col)
    #         _temp = _temp / L.load(i * m + i)
    #         Z.store(i * n + col, _temp)

    #     # Solve `UZ = Z` for `X` for each col
    #     for i in range(m - 1, -1, -1):
    #         var _temp2 = Z.load(i * n + col)
    #         for j in range(i + 1, m):
    #             _temp2 = _temp2 - U.load(i * m + j) * X.load(j * n + col)
    #         _temp2 = _temp2 / U.load(i * m + i)
    #         X.store(i * n + col, _temp2)

    # return X
