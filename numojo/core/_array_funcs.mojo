# from ..traits.NDArrayTraits import NDArrayBackend
from algorithm.functional import parallelize, vectorize, num_physical_cores

"""
Implementing backend for array keeping it simple for now
"""

fn _math_func_1_array_in_one_array_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
        type, simd_w
    ],
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Apply a SIMD function of one variable and one return to a NDArray

    Parameters:
        dtype: The element type.
        func: the SIMD function to to apply.

    Args:
        array: A NDArray

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn closure[simdwidth: Int](i: Int):
        var simd_data = array.load[width=opt_nelts](i)
        result_array.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data)
        )

    vectorize[closure, opt_nelts](array.num_elements())

    return result_array
fn _math_func_2_array_in_one_array_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[type, simd_w],
](
    array1: NDArray[dtype], array2: NDArray[dtype]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD function of two variable and one return to a NDArray

    Constraints:
        Both arrays must have the same shape

    Parameters:
        dtype: The element type.
        func: the SIMD function to to apply.

    Args:
        array1: A NDArray
        array2: A NDArray

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """

    if array1.shape() != array2.shape():
        raise Error(
            "Shape Mismatch error shapes must match for this function"
        )

    var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn closure[simdwidth: Int](i: Int):
        var simd_data1 = array1.load[width=opt_nelts](i)
        var simd_data2 = array2.load[width=opt_nelts](i)
        result_array.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data1, simd_data2)
        )

    vectorize[closure, opt_nelts](result_array.num_elements())
    return result_array

fn _math_func_one_array_one_SIMD_in_one_array_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[type, simd_w],
](
    array: NDArray[dtype], scalar: SIMD[dtype,1]
)  -> NDArray[dtype]:
    """
    Apply a SIMD function of two variable and one return to a NDArray

    Constraints:
        Both arrays must have the same shape

    Parameters:
        dtype: The element type.
        func: the SIMD function to to apply.

    Args:
        array: A NDArray
        scalar: scalar value

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """

    var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn closure[simdwidth: Int](i: Int):
        var simd_data1 = array.load[width=opt_nelts](i)
        result_array.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data1,scalar)
        )

    vectorize[closure, opt_nelts](result_array.num_elements())
    return result_array

