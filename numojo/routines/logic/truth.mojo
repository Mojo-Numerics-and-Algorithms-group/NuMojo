# ===----------------------------------------------------------------------=== #
# Truth value testing
# ===----------------------------------------------------------------------=== #

import math
from algorithm import vectorize, parallelize
from sys import simd_width_of

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix, MatrixBase


fn all[dtype: DType](A: MatrixBase[dtype, **_]) -> Scalar[dtype]:
    """
    Test whether all array elements evaluate to True.

    Args:
        A: Matrix.
    """
    var res = Scalar[dtype](1)
    comptime width: Int = simd_width_of[dtype]()

    @parameter
    fn cal_and[width: Int](i: Int) unified {mut res, read A}:
        res = res & A._buf.ptr.load[width=width](i).reduce_and()

    vectorize[width](A.size, cal_and)
    return res


fn all[
    dtype: DType
](A: MatrixBase[dtype, **_], axis: Int) raises -> Matrix[dtype]:
    """
    Test whether all array elements evaluate to True along axis.
    """

    comptime width: Int = simd_width_of[dtype]()

    if axis == 0:
        var B = Matrix.ones[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int) unified {mut B, read A, read i}:
                B._store[width](
                    0, j, B._load[width](0, j) & A._load[width](i, j)
                )

            vectorize[width](A.shape[1], cal_vec_sum)

        return B^

    elif axis == 1:
        var B = Matrix.ones[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_sum[width: Int](j: Int) unified {mut B, read A, read i}:
                B._store(
                    i,
                    0,
                    B._load(i, 0) & A._load[width=width](i, j).reduce_and(),
                )

            vectorize[width](A.shape[1], cal_sum)

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn all(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    If all True.

    Args:
        array: A NDArray.
    Returns:
        A boolean scalar
    """
    var result = Scalar[DType.bool](True)
    # comptime opt_nelts: Int = simd_width_of[DType.bool]()

    # @parameter
    # fn vectorize_sum[simd_width: Int](idx: Int) -> None:
    #     var simd_data = array.load[width=simd_width](idx)
    #     result |= simd_data.reduce_and()

    # vectorize[vectorize_sum, opt_nelts](array.size)
    # return result
    for i in range(array.size):
        result &= array.load(i)
    return result


fn any(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    If any True.

    Args:
        array: A NDArray.
    Returns:
        A boolean scalar
    """
    var result = Scalar[DType.bool](False)
    # comptime opt_nelts: Int = simd_width_of[DType.bool]()

    # @parameter
    # fn vectorize_sum[simd_width: Int](idx: Int) -> None:
    #     var simd_data = array.load[width=simd_width](idx)
    #     result &= simd_data.reduce_or()

    # vectorize[vectorize_sum, opt_nelts](array.size)
    # return result
    for i in range(array.size):
        result |= array.load(i)
    return result


fn any[dtype: DType](A: MatrixBase[dtype, **_]) -> Scalar[dtype]:
    """
    Test whether any array elements evaluate to True.

    Args:
        A: Matrix.
    """
    var res = Scalar[dtype](0)
    comptime width: Int = simd_width_of[dtype]()

    @parameter
    fn cal_and[width: Int](i: Int) unified {mut res, read A}:
        res = res | A._buf.ptr.load[width=width](i).reduce_or()

    vectorize[width](A.size, cal_and)
    return res


fn any[
    dtype: DType
](A: MatrixBase[dtype, **_], axis: Int) raises -> Matrix[dtype]:
    """
    Test whether any array elements evaluate to True along axis.
    """

    comptime width: Int = simd_width_of[dtype]()

    if axis == 0:
        var B = Matrix.zeros[dtype](shape=(1, A.shape[1]))

        for i in range(A.shape[0]):

            @parameter
            fn cal_vec_sum[width: Int](j: Int) unified {mut B, read A, read i}:
                B._store[width](
                    0, j, B._load[width](0, j) | A._load[width](i, j)
                )

            vectorize[width](A.shape[1], cal_vec_sum)

        return B^

    elif axis == 1:
        var B = Matrix.zeros[dtype](shape=(A.shape[0], 1))

        @parameter
        fn cal_rows(i: Int):
            @parameter
            fn cal_sum[width: Int](j: Int) unified {mut B, read A, read i}:
                B._store(
                    i,
                    0,
                    B._load(i, 0) | A._load[width=width](i, j).reduce_or(),
                )

            vectorize[width](A.shape[1], cal_sum)

        parallelize[cal_rows](A.shape[0], A.shape[0])
        return B^

    else:
        raise Error(String("The axis can either be 1 or 0!"))
