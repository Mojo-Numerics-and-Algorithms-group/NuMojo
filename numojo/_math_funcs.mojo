"""
Implements generic reusable functions for math
"""
from tensor import Tensor
from testing import assert_raises


fn _math_func_1_tensor_in_one_tensor_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
        type, simd_w
    ],
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Apply a SIMD function of one variable and one return to a tensor

    Parameters:
        dtype: The element type.
        func: the SIMD function to to apply.

    Args:
        tensor: A tensor

    Returns:
        A a new tensor that is tensor with the function func applied.
    """
    var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (tensor.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data = tensor.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data)
        )

    if tensor.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (tensor.num_elements() // opt_nelts),
            tensor.num_elements(),
        ):
            var simd_data = func[dtype, 1](tensor.load[width=1](i))
            result_tensor.store[width=1](i, simd_data)
    return result_tensor


fn _math_func_2_tensor_in_one_tensor_out[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[type, simd_w],
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Apply a SIMD function of two variable and one return to a tensor

    Constraints:
        Both tensors must have the same shape

    Parameters:
        dtype: The element type.
        func: the SIMD function to to apply.

    Args:
        tensor1: A tensor
        tensor2: A tensor

    Returns:
        A a new tensor that is tensor with the function func applied.
    """

    if tensor1.shape() != tensor2.shape():
        with assert_raises():
            raise "Shape Mismatch error shapes must match for this function"
    var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (tensor1.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data1 = tensor1.load[width=opt_nelts](i)
        var simd_data2 = tensor2.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data1, simd_data2)
        )

    if tensor1.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (tensor1.num_elements() // opt_nelts),
            tensor1.num_elements(),
        ):
            var simd_data1 = tensor1.load[width=1](i)
            var simd_data2 = tensor2.load[width=1](i)
            result_tensor.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
    return result_tensor


fn _math_func_compare_2_tensors[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[DType.bool, simd_w],
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
    if tensor1.shape() != tensor2.shape():
        with assert_raises():
            raise "Shape Mismatch error shapes must match for this function"
    var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor1.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (tensor1.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data1 = tensor1.load[width=opt_nelts](i)
        var simd_data2 = tensor2.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data1, simd_data2)
        )

    if tensor1.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (tensor1.num_elements() // opt_nelts),
            tensor1.num_elements(),
        ):
            var simd_data1 = tensor1.load[width=1](i)
            var simd_data2 = tensor2.load[width=1](i)
            result_tensor.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
    return result_tensor


fn _math_func_is[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
        DType.bool, simd_w
    ],
](tensor: Tensor[dtype]) -> Tensor[DType.bool]:
    var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (tensor.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data = tensor.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data)
        )

    if tensor.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (tensor.num_elements() // opt_nelts),
            tensor.num_elements(),
        ):
            var simd_data = func[dtype, 1](tensor.load[width=1](i))
            result_tensor.store[width=1](i, simd_data)
    return result_tensor


fn _math_func_simd_int[
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
        type, simd_w
    ],
](tensor1: Tensor[dtype], intval: Int) -> Tensor[dtype]:
    var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (tensor1.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data1 = tensor1.load[width=opt_nelts](i)

        result_tensor.store[width=opt_nelts](
            i, func[dtype, opt_nelts](simd_data1, intval)
        )

    if tensor1.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (tensor1.num_elements() // opt_nelts),
            tensor1.num_elements(),
        ):
            var simd_data1 = tensor1.load[width=1](i)
            result_tensor.store[width=1](i, func[dtype, 1](simd_data1, intval))
    return result_tensor
