from tensor import Tensor

trait Backend:
    """
    A trait that defines backends for calculations in the rest of the library.
    """
    
    fn __init__(inout self:Self):
       pass

    fn _math_func_fma[
        dtype: DType,
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype], tensor3: Tensor[dtype]) raises -> Tensor[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a tensor

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            tensor1: A tensor
            tensor2: A tensor
            tensor3: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        pass
    
    fn _math_func_fma[
        dtype: DType,
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype], simd: SIMD[dtype,1]) raises -> Tensor[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a tensor.

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            tensor1: A tensor.
            tensor2: A tensor.
            simd: A SIMD[dtype,1] value to be added

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        pass

    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[dtype]:
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
        ...


    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
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

        ...

    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
       ...

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[DType.bool]:
        ...


    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor1: Tensor[dtype], intval: Int) -> Tensor[dtype]:
        ...