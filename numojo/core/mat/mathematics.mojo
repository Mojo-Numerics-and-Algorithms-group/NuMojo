"""
`numojo.core.mat.math` module provides mathematical functions for Matrix type.

- Trigonometric functions
- Sums, products, differences

"""

import math

from .matrix import Matrix, _arithmetic_func
from .creation import zeros

# ===-----------------------------------------------------------------------===#
# Trigonometric functions
# ===-----------------------------------------------------------------------===#


fn sin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    return _arithmetic_func[dtype, math.sin](A)


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
    fn cal_sum[width: Int](i: Int):
        res = res + A._buf.load[width=width](i).reduce_add()

    vectorize[cal_sum, width](A.size)
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
            # ===
            # Raw mathod
            # ===
            # for j in range(A.shape[1]):
            #     B._store(i, 0, B._load(i, 0) + A._load(i, j))

            @parameter
            fn cal_sum[width: Int](j: Int):
                B._store(
                    i,
                    0,
                    B._load(i, 0) + A._load[width=width](i, j).reduce_add(),
                )

            vectorize[cal_sum, width](A.shape[1])

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))
