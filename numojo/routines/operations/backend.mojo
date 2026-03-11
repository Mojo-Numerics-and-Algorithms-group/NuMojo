# ===----------------------------------------------------------------------=== #
# NuMojo: Math backend
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Math operations backend (numojo.routines.operations.backend).

Defines vectorized backend structures and reusable SIMD math primitives consumed by the math submodules.
"""

from std.algorithm.functional import vectorize
from std.sys import simd_width_of
from std.builtin.simd import FastMathFlag

from numojo.core.ndarray import NDArray
from numojo.routines.creation import _0darray


# TODO: Add overloads for complexndarray.
# TODO: Add NumojoError as argument so that the calling function can modify the error message with more context.
# NOTE: We currently do all checks within these backend functions,
# but it'll be ideal to have these check done at higher level callers and keep the backend functions clean.
# We will revisit this decision in future.
struct HostExecutor:
    """
    Vectorized CPU Backend.

    This struct provides static methods to apply SIMD-compatible
    unary and binary functions to NDArrays, Scalars.
    """

    fn __init__(out self):
        pass

    @staticmethod
    fn apply_unary[
        dtype: DType,
        simd_width: Int,
        kernel: fn[type: DType, simd_w: Int](SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](scalar: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        """
        Applies a SIMD-compatible unary function to a SIMD value.

        Parameters:
            dtype: The element type.
            simd_width: The SIMD width of the input and output.
            kernel: The SIMD-compatible function to apply.

        Args:
            scalar: The input SIMD value.

        Returns:
            A new SIMD value containing the result of applying the function.
        """
        return kernel[dtype, simd_width](scalar)

    @staticmethod
    fn apply_binary[
        dtype: DType,
        simd_width: Int,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](simd1: SIMD[dtype, simd_width], simd2: SIMD[dtype, simd_width]) -> SIMD[
        dtype, simd_width
    ]:
        """
        Applies a SIMD-compatible binary function to two SIMD values.

        Parameters:
            dtype: The element type.
            simd_width: The SIMD width of the input and output.
            kernel: The SIMD-compatible binary function to apply.

        Args:
            simd1: The first input SIMD value.
            simd2: The second input SIMD value.

        Returns:
            A new SIMD value containing the result of applying the function.
        """
        return kernel[dtype, simd_width](simd1, simd2)

    @staticmethod
    fn apply_unary_predicate[
        dtype: DType,
        simd_width: Int,
        kernel: fn[type: DType, simd_w: Int](SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](simd: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        """
        Applies a SIMD-compatible unary predicate to a SIMD value.

        Parameters:
            dtype: The element type.
            simd_width: The SIMD width of the input and output.
            kernel: The SIMD-compatible unary predicate function to apply.

        Args:
            simd: The input SIMD value.

        Returns:
            A SIMD boolean value containing the predicate result.
        """
        return kernel[dtype, simd_width](simd)

    @staticmethod
    fn apply_binary_predicate[
        dtype: DType,
        simd_width: Int,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](simd1: SIMD[dtype, simd_width], simd2: SIMD[dtype, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        """
        Applies a SIMD-compatible binary predicate to two SIMD values.

        Parameters:
            dtype: The element type.
            simd_width: The SIMD width of the input and output (should be 1 for SIMD).
            kernel: The SIMD-compatible binary predicate function to apply.

        Args:
            simd1: The first input SIMD value.
            simd2: The second input SIMD value.

        Returns:
            A SIMD boolean value containing the predicate result.
        """
        return kernel[dtype, simd_width](simd1, simd2)

    @staticmethod
    fn apply_unary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](array: NDArray[dtype]) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible unary function to an NDArray.

        Parameters:
            dtype: The element type of the NDArray.
            kernel: The SIMD-compatible function to apply.

        Args:
            array: The input NDArray.

        Returns:
            A new NDArray containing the result of applying the function.
        """
        # View safety guard: ensure input is C-contiguous before SIMD access.
        if not array.is_c_contiguous():
            return Self.apply_unary[dtype, kernel](array.contiguous())

        # For 0darray (numojo scalar)
        # Treat it as a scalar and apply the function
        if array.ndim == 0:
            var result_array = _0darray(
                val=kernel[dtype, 1]((array._buf.ptr + array.offset)[])
            )
            return result_array^

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[simd_w: Int](i: Int) unified {mut result_array, read array}:
            var simd_data = array._buf.ptr.load[width=simd_w](i)
            result_array._buf.ptr.store(i, kernel[dtype, simd_w](simd_data))

        vectorize[width](array.size, closure)

        return result_array^

    @staticmethod
    fn apply_binary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible binary function to two NDArrays.

        Parameters:
            dtype: The element type of the NDArrays.
            kernel: The SIMD-compatible binary function to apply.

        Args:
            array1: The first input NDArray.
            array2: The second input NDArray.

        Returns:
            A new NDArray containing the result of applying the function.
        """
        if not array1.is_c_contiguous() and not array2.is_c_contiguous():
            return Self.apply_binary[dtype, kernel](
                array1.contiguous(), array2.contiguous()
            )

        if not array1.is_c_contiguous():
            return Self.apply_binary[dtype, kernel](array1.contiguous(), array2)
        if not array2.is_c_contiguous():
            return Self.apply_binary[dtype, kernel](array1, array2.contiguous())

        # For 0darray (numojo scalar)
        # Treat it as a scalar and apply the function
        if array1.ndim == 0:
            return Self.apply_binary[dtype, kernel](array1[], array2)
        if array2.ndim == 0:
            return Self.apply_binary[dtype, kernel](array1, array2[])

        if array1.shape != array2.shape:
            raise Error(
                "Shape Mismatch error: shapes must match for this function"
            )

        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[
            simd_w: Int
        ](i: Int) unified {mut result_array, read array1, read array2}:
            var simd_data1 = array1._buf.ptr.load[width=simd_w](i)
            var simd_data2 = array2._buf.ptr.load[width=simd_w](i)
            result_array._buf.ptr.store(
                i, kernel[dtype, simd_w](simd_data1, simd_data2)
            )

        vectorize[width](result_array.size, closure)
        return result_array^

    @staticmethod
    fn apply_binary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](array: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible binary function to an NDArray and a scalar.

        Parameters:
            dtype: The element type of the NDArray.
            kernel: The SIMD-compatible binary function to apply.

        Args:
            array: The input NDArray.
            scalar: The input scalar value.

        Returns:
            A new NDArray containing the result of applying the function.
        """
        # View safety guard: ensure input is C-contiguous before SIMD access.
        if not array.is_c_contiguous():
            return Self.apply_binary[dtype, kernel](array.contiguous(), scalar)

        # For 0darray (numojo scalar)
        # Treat it as a scalar and apply the function
        if array.ndim == 0:
            var result_array = _0darray(val=kernel[dtype, 1](array[], scalar))
            return result_array^

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[
            simd_w: Int
        ](i: Int) unified {mut result_array, read array, read scalar}:
            var simd_data1 = array._buf.ptr.load[width=simd_w](i)
            result_array._buf.ptr.store(
                i, kernel[dtype, simd_w](simd_data1, scalar)
            )

        vectorize[width](result_array.size, closure)
        return result_array^

    @staticmethod
    fn apply_binary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](scalar: SIMD[dtype, 1], array: NDArray[dtype]) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible binary function to a scalar and an NDArray.

        Parameters:
            dtype: The element type of the NDArray.
            kernel: The SIMD-compatible binary function to apply.

        Args:
            scalar: The input scalar value.
            array: The input NDArray.

        Returns:
            A new NDArray containing the result of applying the function.
        """

        # View safety guard: ensure input is C-contiguous before SIMD access.
        if not array.is_c_contiguous():
            return Self.apply_binary[dtype, kernel](scalar, array.contiguous())

        # For 0darray (numojo scalar)
        # Treat it as a scalar and apply the function
        if array.ndim == 0:
            var result_array = _0darray(val=kernel[dtype, 1](scalar, array[]))
            return result_array^

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[
            simd_w: Int
        ](i: Int) unified {mut result_array, read array, read scalar}:
            var simd_data1 = array._buf.ptr.load[width=simd_w](i)
            result_array._buf.ptr.store(
                i, kernel[dtype, simd_w](scalar, simd_data1)
            )

        vectorize[width](result_array.size, closure)
        return result_array^

    @staticmethod
    fn apply_binary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible binary function to an NDArray and an Int scalar.

        Parameters:
            dtype: The element type of the NDArray.
            kernel: The SIMD-compatible binary function to apply.

        Args:
            array: The input NDArray.
            intval: The input integer value.

        Returns:
            A new NDArray containing the result of applying the function.
        """
        # View safety guard: ensure input is C-contiguous before SIMD access.
        if not array.is_c_contiguous():
            return Self.apply_binary[dtype, kernel](array.contiguous(), intval)

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[
            simd_w: Int
        ](i: Int) unified {mut result_array, read array, read intval}:
            var simd_data = array._buf.ptr.load[width=simd_w](i)

            result_array._buf.ptr.store(
                i, kernel[dtype, simd_w](simd_data, intval)
            )

        vectorize[width](array.size, closure)
        return result_array^

    @staticmethod
    fn apply_binary_predicate[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
        DType.bool
    ]:
        """
        Applies a SIMD-compatible binary predicate to two NDArrays, returning a boolean NDArray.

        Parameters:
            dtype: The element type of the input NDArrays.
            kernel: The SIMD-compatible binary predicate function to apply.

        Args:
            array1: The first input NDArray.
            array2: The second input NDArray.

        Returns:
            A new boolean NDArray containing the result of the predicate.
        """
        if not array1.is_c_contiguous() and not array2.is_c_contiguous():
            return Self.apply_binary_predicate[dtype, kernel](
                array1.contiguous(), array2.contiguous()
            )

        # View safety guard: ensure inputs are C-contiguous before SIMD access.
        if not array1.is_c_contiguous():
            return Self.apply_binary_predicate[dtype, kernel](
                array1.contiguous(), array2
            )
        if not array2.is_c_contiguous():
            return Self.apply_binary_predicate[dtype, kernel](
                array1, array2.contiguous()
            )

        # For 0darray (numojo scalar)
        # Treat it as a scalar and apply the function
        if array2.ndim == 0:
            return Self.apply_binary_predicate[dtype, kernel](array1, array2[])

        if array1.shape != array2.shape:
            raise Error(
                "Shape Mismatch error: shapes must match for this function"
            )

        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape
        )
        comptime width = simd_width_of[DType.bool]()

        @parameter
        fn closure[
            simd_w: Int
        ](i: Int) unified {mut result_array, read array1, read array2}:
            var simd_data1 = array1._buf.ptr.load[width=simd_w](i)
            var simd_data2 = array2._buf.ptr.load[width=simd_w](i)

            bool_simd_store[simd_w](
                result_array._buf.ptr,
                i,
                kernel[dtype, simd_w](simd_data1, simd_data2),
            )

        vectorize[width](array1.size, closure)
        return result_array^

    @staticmethod
    fn apply_binary_predicate[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](array1: NDArray[dtype], scalar: SIMD[dtype, 1]) raises -> NDArray[
        DType.bool
    ]:
        """
        Applies a SIMD-compatible binary predicate to an NDArray and a scalar, returning a boolean NDArray.

        Parameters:
            dtype: The element type of the input NDArray.
            kernel: The SIMD-compatible binary predicate function to apply.

        Args:
            array1: The input NDArray.
            scalar: The input scalar value.

        Returns:
            A new boolean NDArray containing the result of the predicate.
        """
        # View safety guard: ensure input is C-contiguous before SIMD access.
        if not array1.is_c_contiguous():
            return Self.apply_binary_predicate[dtype, kernel](
                array1.contiguous(), scalar
            )

        # For 0darray (numojo scalar)
        # Treat it as a scalar and apply the function
        if array1.ndim == 0:
            var result_array = _0darray(val=kernel[dtype, 1](array1[], scalar))
            return result_array^

        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape
        )
        comptime width = simd_width_of[DType.bool]()

        @parameter
        fn closure[
            simd_w: Int
        ](i: Int) unified {mut result_array, read array1, read scalar}:
            var simd_data1 = array1._buf.ptr.load[width=simd_w](i)
            var simd_data2 = SIMD[dtype, simd_w](scalar)
            bool_simd_store[simd_w](
                result_array.unsafe_ptr(),
                i,
                kernel[dtype, simd_w](simd_data1, simd_data2),
            )

        vectorize[width](array1.size, closure)
        return result_array^

    @staticmethod
    fn apply_unary_predicate[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Applies a SIMD-compatible unary predicate to an NDArray, returning a boolean NDArray.

        Parameters:
            dtype: The element type of the input NDArray.
            kernel: The SIMD-compatible unary predicate function to apply.

        Args:
            array: The input NDArray.

        Returns:
            A new boolean NDArray containing the result of the predicate.
        """
        # View safety guard: ensure input is C-contiguous before SIMD access.
        if not array.is_c_contiguous():
            return Self.apply_unary_predicate[dtype, kernel](array.contiguous())

        var result_array: NDArray[DType.bool] = NDArray[DType.bool](array.shape)
        comptime width = simd_width_of[DType.bool]()

        @parameter
        fn closure[simd_w: Int](i: Int) unified {mut result_array, read array}:
            var simd_data = array._buf.ptr.load[width=simd_w](i)
            bool_simd_store[simd_w](
                result_array._buf.ptr,
                i,
                kernel[dtype, simd_w](simd_data),
            )

        vectorize[width](array.size, closure)
        return result_array^

    @staticmethod
    fn apply_ternary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible ternary function to three NDArrays.

        Parameters:
            dtype: The element type of the NDArrays.
            kernel: The SIMD-compatible ternary function to apply.

        Args:
            array1: The first input NDArray.
            array2: The second input NDArray.
            array3: The third input NDArray.

        Returns:
            A new NDArray containing the result of applying the function.
        """
        if (
            not array1.is_c_contiguous()
            and not array2.is_c_contiguous()
            and not array3.is_c_contiguous()
        ):
            return Self.apply_ternary[dtype, kernel](
                array1.contiguous(), array2.contiguous(), array3.contiguous()
            )

        if not array1.is_c_contiguous():
            return Self.apply_ternary[dtype, kernel](
                array1.contiguous(), array2, array3
            )
        if not array2.is_c_contiguous():
            return Self.apply_ternary[dtype, kernel](
                array1, array2.contiguous(), array3
            )
        if not array3.is_c_contiguous():
            return Self.apply_ternary[dtype, kernel](
                array1, array2, array3.contiguous()
            )

        if array1.shape != array2.shape and array1.shape != array3.shape:
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )

        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[
            simdwidth: Int
        ](i: Int) unified {
            mut result_array, read array1, read array2, read array3
        }:
            var simd_data1 = array1._buf.ptr.load[width=simdwidth](i)
            var simd_data2 = array2._buf.ptr.load[width=simdwidth](i)
            var simd_data3 = array3._buf.ptr.load[width=simdwidth](i)
            result_array._buf.ptr.store(
                i, kernel(simd_data1, simd_data2, simd_data3)
            )

        vectorize[width](array1.size, closure)
        return result_array^

    @staticmethod
    fn apply_ternary[
        dtype: DType,
        kernel: fn[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        array1: NDArray[dtype], array2: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[dtype]:
        """
        Applies a SIMD-compatible ternary function to two NDArrays and a scalar.

        Parameters:
            dtype: The element type of the input NDArrays.
            kernel: The SIMD-compatible ternary function to apply.

        Args:
            array1: The first input NDArray.
            array2: The second input NDArray.
            scalar: The input scalar value.

        Returns:
            A new NDArray containing the result of applying the function.
        """
        if not array1.is_c_contiguous() and not array2.is_c_contiguous():
            return Self.apply_ternary[dtype, kernel](
                array1.contiguous(), array2.contiguous(), scalar
            )

        if not array1.is_c_contiguous():
            return Self.apply_ternary[dtype, kernel](
                array1.contiguous(), array2, scalar
            )
        if not array2.is_c_contiguous():
            return Self.apply_ternary[dtype, kernel](
                array1, array2.contiguous(), scalar
            )

        if array1.shape != array2.shape:
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )

        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape)
        comptime width = simd_width_of[dtype]()

        @parameter
        fn closure[
            simdwidth: Int
        ](i: Int) unified {
            mut result_array, read array1, read array2, read scalar
        }:
            var simd_data1 = array1._buf.ptr.load[width=simdwidth](i)
            var simd_data2 = array2._buf.ptr.load[width=simdwidth](i)
            result_array._buf.ptr.store(
                i, kernel(simd_data1, simd_data2, scalar)
            )

        vectorize[width](array1.size, closure)
        return result_array^


# This provides a way to bypass bitpacking issues with Bool
fn bool_simd_store[
    ptr_origin: MutOrigin,
    //,
    simd_width: Int,
](
    ptr: UnsafePointer[Scalar[DType.bool], ptr_origin],
    start: Int,
    val: SIMD[DType.bool, simd_width],
):
    """
    Workaround function for storing bools from a SIMD vector into an UnsafePointer.

    Parameters:
        ptr_origin: Origin of the pointer.
        simd_width: The SIMD width of the stored value.

    Args:
        ptr: Pointer to be written to.
        start: Start position in the pointer.
        val: SIMD boolean value to store.
    """
    (ptr + start).strided_store[width=simd_width](val=val, stride=1)
