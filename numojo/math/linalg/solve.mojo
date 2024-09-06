from ...core.ndarray import NDArray

fn lu_decomposition[dtype: DType = DType.float64](array: NDArray) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
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

    Reference:  
        Linear Algebra And Its Applications, fourth edition, Gilbert Strang  
        https://en.wikipedia.org/wiki/LU_decomposition  
        https://www.scicoding.com/how-to-calculate-lu-decomposition-in-python/
    """

    # Check whether the dimension is 2
    if array.ndim != 2:
        raise("The array is not 2-dimensional!")
    
    # Check whether the matrix is square
    var shape_of_array = array.shape()
    var m = shape_of_array[0]
    var n = shape_of_array[1]
    if m != n:
        raise("The matrix is not square!")
    
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
                L.__setitem__(List[Int](j, i), (A.item(j, i) - sum_of_products_for_L) / U.item(i, i))

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L.item(i, k) * U.item(k, j)
            U.__setitem__(List[Int](i, j), A.item(i, j) - sum_of_products_for_U)

    return L, U






