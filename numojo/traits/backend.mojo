# ===----------------------------------------------------------------------=== #
# Defines computational backend traits
# ===----------------------------------------------------------------------=== #


from ..core.ndarray import NDArray


trait Backend:
    """
    A trait that defines backends for calculations in the rest of the library.
    """

    fn __init__(mut self: Self):
        """
        Initialize the backend.
        """
        pass

    fn math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

        Constraints:
            Both arrays must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            array3: A NDArray.

        Returns:
            A a new NDArray that is NDArray with the function func applied.

        Raises:
            If shapes are missmatched or there is a access error.
        """
        pass

    fn math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        simd: SIMD[dtype, 1],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

        Constraints:
            Both arrays must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added.

        Returns:
            A new NDArray that is NDArray with the function func applied.
        """
        pass

    fn math_func_1_array_in_one_array_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of one variable and one return to a NDArray.

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.

        Returns:
            A new NDArray that is NDArray with the function func applied.
        """
        ...

    fn math_func_2_array_in_one_array_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray.

        Constraints:
            Both arrays must have the same shape

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array1: A NDArray.
            array2: A NDArray.

        Returns:
            A new NDArray that is NDArray with the function func applied.
        """

        ...

    fn math_func_1_array_1_scalar_in_one_array_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array: NDArray[dtype], scalar: Scalar[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray.

        Constraints:
            Both arrays must have the same shape

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A new NDArray that is NDArray with the function func applied.
        """

        ...

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        """
        Apply a SIMD comparision function of two variable.

        Constraints:
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.
            func: The SIMD comparision function to to apply.

        Args:
            array1: A NDArray.
            array2: A NDArray.

        Returns:
            A new Boolean NDArray that is NDArray with the function func applied.
        """
        ...

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        """
        Apply a SIMD comparision function of two variable.

        Constraints:
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.
            func: The SIMD comparision function to to apply.

        Args:
            array1: A NDArray.
            scalar: A scalar.

        Returns:
            A new Boolean NDArray that is NDArray with the function func applied.
        """
        ...

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        ...

    # fn math_func_simd_int[
    #     dtype: DType,
    #     func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
    #         type, simd_w
    #     ],
    # ](self: Self, array1: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
    #     ...
