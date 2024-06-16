# ===----------------------------------------------------------------------=== #
# Defines computational backend traits
# ===----------------------------------------------------------------------=== #


from ..ndarray import NDArray


trait Backend:
    """
    A trait that defines backends for calculations in the rest of the library.
    """

    fn __init__(inout self: Self):
        """
        Initialize the backend.
        """
        pass

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        tensor1: NDArray[dtype],
        tensor2: NDArray[dtype],
        tensor3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            tensor1: A NDArray
            tensor2: A NDArray
            tensor3: A NDArray

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        pass

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        tensor1: NDArray[dtype],
        tensor2: NDArray[dtype],
        simd: SIMD[dtype, 1],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            tensor1: A NDArray.
            tensor2: A NDArray.
            simd: A SIMD[dtype,1] value to be added

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        pass

    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[dtype]:
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
        ...

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, tensor1: NDArray[dtype], tensor2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor1: A NDArray
            tensor2: A NDArray

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        ...

    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, tensor1: NDArray[dtype], tensor2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        ...

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        ...

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, tensor1: NDArray[dtype], intval: Int) -> NDArray[dtype]:
        ...
