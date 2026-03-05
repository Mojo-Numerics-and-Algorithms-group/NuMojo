# ===----------------------------------------------------------------------=== #
# NuMojo: Truth testing
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Truth value testing (numojo.routines.logic.truth)

This module implements the truth value testing functions, such as `all` and `any`, for both `NDArray` and `Matrix`.
"""

from algorithm import vectorize, parallelize
from sys import simd_width_of

from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix

# TODO: Add all and any algorithm to backend.

# ===----------------------------------------------------------------------=== #
# Truth operations for NDArray
# ===----------------------------------------------------------------------=== #


fn all(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    Checks whether all elements of the array evaluate to True.

    Args:
        array: Input NDArray (DType.bool).

    Returns:
        True if all elements of the array evaluate to True, False if not.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.truth import all

        var a = arange[i32](24).reshape(Shape(2, 3, 4))
        var result = all(a > 5) # outputs False
        ```
    """
    var result = Scalar[DType.bool](True)
    comptime opt_nelts: Int = simd_width_of[DType.bool]()

    @parameter
    fn closure[
        simd_width: Int
    ](idx: Int) unified {mut result, read array} -> None:
        var simd_data = array.unsafe_load[width=simd_width](idx)
        result = (result & simd_data).reduce_and()

    vectorize[opt_nelts](array.size, closure)
    return result


fn any(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    Checks whether any element of the array evaluate to True.

    Args:
        array: Input NDArray (DType.bool).

    Returns:
        True if any element of the array evaluate to True, False if not.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.truth import any

        var a = arange[i32](24).reshape(Shape(2, 3, 4))
        var result = any(a > 5) # outputs True
        ```
    """
    var result = Scalar[DType.bool](False)
    comptime opt_nelts: Int = simd_width_of[DType.bool]()

    @parameter
    fn closure[
        simd_width: Int
    ](idx: Int) unified {mut result, read array} -> None:
        var simd_data = array.unsafe_load[width=simd_width](idx)
        result = (result | simd_data).reduce_or()

    vectorize[opt_nelts](array.size, closure)
    return result


# ===----------------------------------------------------------------------=== #
# Truth operations for Matrix
# ===----------------------------------------------------------------------=== #


fn all[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Test whether all array elements evaluate to True.

    Args:
        A: Matrix.
    """
    if not A.is_c_contiguous():
        return all(A.contiguous())

    var res = Scalar[dtype](1)
    comptime width: Int = simd_width_of[dtype]()

    @parameter
    fn closure[width: Int](i: Int) unified {mut res, read A}:
        res = (res & A._buf.ptr.load[width=width](i)).reduce_and()

    vectorize[width](A.size, closure)
    return res


fn all[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
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


fn any[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Test whether any array elements evaluate to True.

    Args:
        A: Matrix.
    """
    if not A.is_c_contiguous():
        return any(A.contiguous())
    var res = Scalar[dtype](0)
    comptime width: Int = simd_width_of[dtype]()

    @parameter
    fn cal_and[width: Int](i: Int) unified {mut res, read A}:
        res = res | A._buf.ptr.load[width=width](i).reduce_or()

    vectorize[width](A.size, cal_and)
    return res


fn any[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
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
