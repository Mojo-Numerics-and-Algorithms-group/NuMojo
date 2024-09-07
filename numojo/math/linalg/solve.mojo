"""
Linear Algebra Solver

Provides:
    - Solver of `Ax = y` using LU decomposition algorithm.
    - Inverse of an invertible matrix.
"""

from ...core.ndarray import NDArray


fn lu_decomposition[
    dtype: DType = DType.float64
](array: NDArray) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Perform LU (lower-upper) decomposition for matrix.

    Parameters:
        dtype: Data type of the upper and upper triangular matrices.

    Args:
        array: Input matrix for decoposition.

    Returns:
        A tuple of the upper and lower triangular matrices.

    Example:
    ```mojo
    import numojo as nm
    fn main() raises:
        var arr = nm.NDArray[nm.f64]("[[1,2,3], [4,5,6], [7,8,9]]")
        var U: nm.NDArray
        var L: nm.NDArray
        L, U = nm.math.linalg.solve.lu_decomposition(arr)
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
    if array.ndim != 2:
        raise ("The array is not 2-dimensional!")

    # Check whether the matrix is square
    var shape_of_array = array.shape()
    var m = shape_of_array[0]
    var n = shape_of_array[1]
    if m != n:
        raise ("The matrix is not square!")

    # Check whether the matrix is singular
    # if singular:
    #     raise("The matrix is singular!")

    # Change dtype of array to defined dtype
    var A = array.astype[dtype]()

    # Initiate upper and lower triangular matrices
    var U = NDArray[dtype](shape=shape_of_array, fill=0)
    var L = NDArray[dtype](shape=shape_of_array, fill=0)

    # Fill in L and U
    for i in range(0, n):
        for j in range(i, n):
            # Fill in L
            if i == j:
                L.__setitem__(List[Int](i, i), 1)
            else:
                var sum_of_products_for_L: Scalar[dtype] = 0
                for k in range(0, i):
                    sum_of_products_for_L += L.item(j, k) * U.item(k, i)
                L.__setitem__(
                    List[Int](j, i),
                    (A.item(j, i) - sum_of_products_for_L) / U.item(i, i),
                )

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L.item(i, k) * U.item(k, j)
            U.__setitem__(List[Int](i, j), A.item(i, j) - sum_of_products_for_U)

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
    var x = NDArray[dtype](m, fill=0)

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
    var x = NDArray[dtype](m, fill=0)

    for i in range(m - 1, -1, -1):
        var value_on_hold: Scalar[dtype] = y.item(i)
        for j in range(i + 1, m):
            value_on_hold = value_on_hold - U.item(i, j) * x.item(j)
        value_on_hold = value_on_hold / U.item(i, i)
        x.__setitem__(i, value_on_hold)

    return x


fn inverse[
    dtype: DType = DType.float64
](array: NDArray) raises -> NDArray[dtype]:
    """Find the inverse of a non-singular, square matrix.

    Parameters:
        dtype: Data type of the inversed matrix. Default value is `f64`.

    Args:
        array: Input matrix. It should be non-singular and square.

    Returns:
        The reversed matrix of the original matrix.

    An example goes as follows:

    ```mojo
    import numojo as nm
    fn main() raises:
        var A = nm.NDArray("[[1,0,1], [0,2,1], [1,1,1]]")
        var B = nm.math.linalg.solve.inverse(A)
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

    var U: NDArray
    var L: NDArray
    L, U = lu_decomposition(array)

    var m = array.shape()[0]
    var inversed = NDArray[dtype](shape=array.shape())

    # Initialize vectors
    var y = NDArray(m)
    var z = NDArray(m)
    var x = NDArray(m)

    for i in range(m):
        # Each time, one of the item is changed to 1
        y = NDArray(m, fill=0)
        y.__setitem__(i, 1)

        # Solve `Lz = y` for `z`
        z = forward_substitution(L, y)

        # Solve `Ux = z` for `x`
        x = back_substitution(U, z)

        for j in range(m):
            inversed.__setitem__(List[Int](j, i), x.item(j))

    return inversed
