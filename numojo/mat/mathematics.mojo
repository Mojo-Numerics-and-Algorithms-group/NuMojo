"""
`numojo.mat.mathematics` module provides mathematical functions for Matrix type.

- Trigonometric functions
- Sums, products, differences

"""

import math
import builtin

from .matrix import Matrix, _arithmetic_func_matrix_to_matrix
from .creation import zeros
from .linalg import transpose

# ===-----------------------------------------------------------------------===#
# Trigonometric functions
# ===-----------------------------------------------------------------------===#


fn sin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.sin](A)


fn cos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.cos](A)


fn tan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.tan](A)


fn arcsin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.asin](A)


fn asin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.asin](A)


fn arccos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.acos](A)


fn acos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.acos](A)


fn arctan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.atan](A)


fn atan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.atan](A)


# ===-----------------------------------------------------------------------===#
# Hyperbolic functions
# ===-----------------------------------------------------------------------===#


fn sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.sinh](A)


fn cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.cosh](A)


fn tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.tanh](A)


fn arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.asinh](A)


fn asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.asinh](A)


fn arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.acosh](A)


fn acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.acosh](A)


fn arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.atanh](A)


fn atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func_matrix_to_matrix[dtype, math.atanh](A)


# ===-----------------------------------------------------------------------===#
# Rounding
# ===-----------------------------------------------------------------------===#


fn round[
    dtype: DType
](owned A: Matrix[dtype], decimals: Int = 0) -> Matrix[dtype]:
    # FIXME
    # The built-in `round` function is not working now.
    # It will be fixed in future.

    for i in range(A.size):
        A._buf[i] = builtin.math.round(A._buf[i], ndigits=decimals)

    return A^


# ===-----------------------------------------------------------------------===#
# Sums, products, differences
# ===-----------------------------------------------------------------------===#


fn sum[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Sum up all items in the Matrix.

    Args:
        A: Matrix.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.sum(A))
    ```
    """
    var res = Scalar[dtype](0)
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn cal_vec[width: Int](i: Int):
        res = res + A._buf.load[width=width](i).reduce_add()

    vectorize[cal_vec, width](A.size)
    return res


fn sum[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Sum up the items in a Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.sum(A, axis=0))
    print(mat.sum(A, axis=1))
    ```
    """

    alias width: Int = simdwidthof[dtype]()

    if axis == 0:
        var B = zeros[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int):
                B._store[width](
                    0, j, B._load[width](0, j) + A._load[width](i, j)
                )

            vectorize[cal_vec_sum, width](A.shape[1])

        return B^

    elif axis == 1:
        var B = zeros[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_vec[width: Int](j: Int):
                B._store(
                    i,
                    0,
                    B._load(i, 0) + A._load[width=width](i, j).reduce_add(),
                )

            vectorize[cal_vec, width](A.shape[1])

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn prod[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Product of all items in the Matrix.

    Args:
        A: Matrix.
    """
    var res = Scalar[dtype](1)
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn cal_vec[width: Int](i: Int):
        res = res * A._buf.load[width=width](i).reduce_mul()

    vectorize[cal_vec, width](A.size)
    return res


fn prod[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Product of items in a Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.prod(A, axis=0))
    print(mat.prod(A, axis=1))
    ```
    """

    alias width: Int = simdwidthof[dtype]()

    if axis == 0:
        var B = ones[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int):
                B._store[width](
                    0, j, B._load[width](0, j) * A._load[width](i, j)
                )

            vectorize[cal_vec_sum, width](A.shape[1])

        return B^

    elif axis == 1:
        var B = ones[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_vec[width: Int](j: Int):
                B._store(
                    i,
                    0,
                    B._load(i, 0) * A._load[width=width](i, j).reduce_mul(),
                )

            vectorize[cal_vec, width](A.shape[1])

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn cumsum[dtype: DType](owned A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Cumsum of flattened matrix.

    Args:
        A: Matrix.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.cumsum(A))
    ```
    """

    A.resize(shape=(1, A.size))

    for i in range(1, A.size):
        A._buf[i] += A._buf[i - 1]

    return A^


fn cumsum[
    dtype: DType
](owned A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Cumsum of Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.cumsum(A, axis=0))
    print(mat.cumsum(A, axis=1))
    ```
    """

    alias width: Int = simdwidthof[dtype]()

    if axis == 0:
        for i in range(1, A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int):
                A._store[width](
                    i, j, A._load[width](i - 1, j) + A._load[width](i, j)
                )

            vectorize[cal_vec_sum, width](A.shape[1])

        return A^

    elif axis == 1:
        return transpose(cumsum(transpose(A), axis=0))

    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn cumprod[dtype: DType](owned A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Cumprod of flattened matrix.

    Args:
        A: Matrix.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.cumprod(A))
    ```
    """

    A.resize(shape=(1, A.size))

    for i in range(1, A.size):
        A._buf[i] *= A._buf[i - 1]

    return A^


fn cumprod[
    dtype: DType
](owned A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Cumprod of Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(100, 100))
    print(mat.cumprod(A, axis=0))
    print(mat.cumprod(A, axis=1))
    ```
    """

    alias width: Int = simdwidthof[dtype]()

    if axis == 0:
        for i in range(1, A.shape[0]):

            @parameter
            fn cal_vec[width: Int](j: Int):
                A._store[width](
                    i, j, A._load[width](i - 1, j) * A._load[width](i, j)
                )

            vectorize[cal_vec, width](A.shape[1])

        return A^

    elif axis == 1:
        return transpose(cumprod(transpose(A), axis=0))

    else:
        raise Error(String("The axis can either be 1 or 0!"))
