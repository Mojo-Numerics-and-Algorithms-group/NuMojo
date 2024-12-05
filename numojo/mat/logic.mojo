"""
`numojo.mat.logic` module provides logic functions for Matrix type.

- Truth value testing

"""

from .matrix import Matrix
from .creation import zeros, ones

# ===-----------------------------------------------------------------------===#
# Truth value testing
# ===-----------------------------------------------------------------------===#


fn all[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Test whether all array elements evaluate to True.

    Args:
        A: Matrix.
    """
    var res = Scalar[dtype](1)
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn cal_and[width: Int](i: Int):
        res = res & A._buf.load[width=width](i).reduce_and()

    vectorize[cal_and, width](A.size)
    return res


fn all[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Test whether all array elements evaluate to True along axis.
    """

    alias width: Int = simdwidthof[dtype]()

    if axis == 0:
        var B = ones[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int):
                B._store[width](
                    0, j, B._load[width](0, j) & A._load[width](i, j)
                )

            vectorize[cal_vec_sum, width](A.shape[1])

        return B^

    elif axis == 1:
        var B = ones[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_sum[width: Int](j: Int):
                B._store(
                    i,
                    0,
                    B._load(i, 0) & A._load[width=width](i, j).reduce_and(),
                )

            vectorize[cal_sum, width](A.shape[1])

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn any[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Test whether any array elements evaluate to True.

    Args:
        A: Matrix.
    """
    var res = Scalar[dtype](0)
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn cal_and[width: Int](i: Int):
        res = res | A._buf.load[width=width](i).reduce_or()

    vectorize[cal_and, width](A.size)
    return res


fn any[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Test whether any array elements evaluate to True along axis.
    """

    alias width: Int = simdwidthof[dtype]()

    if axis == 0:
        var B = zeros[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int):
                B._store[width](
                    0, j, B._load[width](0, j) | A._load[width](i, j)
                )

            vectorize[cal_vec_sum, width](A.shape[1])

        return B^

    elif axis == 1:
        var B = zeros[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_sum[width: Int](j: Int):
                B._store(
                    i,
                    0,
                    B._load(i, 0) | A._load[width=width](i, j).reduce_or(),
                )

            vectorize[cal_sum, width](A.shape[1])

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))
