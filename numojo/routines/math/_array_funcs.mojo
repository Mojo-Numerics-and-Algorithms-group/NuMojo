"""
Implementing backend for array keeping it simple for now
"""
# from ..traits.NDArrayTraits import NDArrayBackend
from algorithm.functional import parallelize, vectorize
from sys.info import num_physical_cores
from sys import simd_width_of

from numojo.core.ndarray import NDArray


fn math_func_1_array_in_one_array_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
        type, simd_w
    ],
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply a SIMD compatible function to a NDArray and returns a new NDArray.

    Parameters:
        dtype: The NDArray element type.
        func: The SIMD compatible function to act on the NDArray.

    Args:
        array: A NDArray.

    Returns:
        A new NDArray that is the result of applying the function to the NDArray.
    """
    var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
    comptime width = simd_width_of[dtype]()

    @parameter
    fn closure[simd_width: Int](i: Int) unified {mut result_array, read array}:
        var simd_data = array._buf.ptr.load[width=simd_width](i)
        result_array._buf.ptr.store(i, func[dtype, simd_width](simd_data))

    vectorize[width](array.size, closure)

    return result_array^


fn math_func_2_array_in_one_array_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[type, simd_w],
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply a SIMD compatible function to two NDArrays and returns a new NDArray.

    Raises:
        Error if the two arrays do not have the same shape.

    Parameters:
        dtype: The NDArray element type.
        func: The SIMD compatible function to act on the NDArrays.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A new NDArray that is the result of applying the function to the input NDArrays.
    """

    if array1.shape != array2.shape:
        raise Error("Shape Mismatch error shapes must match for this function")

    var result_array: NDArray[dtype] = NDArray[dtype](array1.shape)
    comptime width = simd_width_of[dtype]()

    @parameter
    fn closure[
        simd_width: Int
    ](i: Int) unified {mut result_array, read array1, read array2}:
        var simd_data1 = array1._buf.ptr.load[width=simd_width](i)
        var simd_data2 = array2._buf.ptr.load[width=simd_width](i)
        result_array._buf.ptr.store(
            i, func[dtype, simd_width](simd_data1, simd_data2)
        )

    vectorize[width](result_array.size, closure)

    return result_array^


fn math_func_one_array_one_SIMD_in_one_array_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[type, simd_w],
](array: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[dtype]:
    """
    Apply a SIMD compatible function to a NDArray and a SIMD value and returns a new NDArray.

    Parameters:
        dtype: The NDArray element type.
        func: The SIMD compatible function to act on the NDArray and SIMD value.

    Args:
        array: A NDArray.
        scalar: A scalar value.

    Returns:
        A new NDArray that is the result of applying the function to the input NDArray and SIMD value.
    """

    var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
    comptime width = simd_width_of[dtype]()

    @parameter
    fn closure[
        simd_width: Int
    ](i: Int) unified {mut result_array, read array, read scalar}:
        var simd_data1 = array._buf.ptr.load[width=simd_width](i)
        result_array._buf.ptr.store(
            i, func[dtype, simd_width](simd_data1, scalar)
        )

    vectorize[width](result_array.size, closure)
    return result_array^
