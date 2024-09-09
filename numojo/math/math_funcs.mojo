"""
Implements backend functions for mathematics
"""
# ===----------------------------------------------------------------------=== #
# Implements generic reusable functions for math
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #


from testing import assert_raises
from algorithm.functional import parallelize, vectorize, num_physical_cores

from ..traits.backend import Backend
from ..core.ndarray import NDArray, NDArrayShape

# TODO Add string method to give name


struct Vectorized(Backend):
    """
    Vectorized Backend Struct.
    Parameters
        unroll_factor: factor by which loops are unrolled.

    Defualt Numojo computation backend takes advantage of SIMD.
    Uses defualt simdwidth.
    """

    fn __init__(inout self: Self):
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
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        # var op_count:Int =0
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            var simd_data3 = array3.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )
            # op_count+=1

        vectorize[closure, width](array1.num_elements())
        # print(op_count)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            # var simd_data3 = array3.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[closure, width](array1.num_elements())
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data)
            )

        vectorize[closure, width](array.num_elements())

        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )

        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data1, simd_data2)
            )

        vectorize[closure, width](result_array.num_elements())
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array.load[width=simdwidth](i)
            var simd_data2 = scalar
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data1, simd_data2)
            )

        vectorize[closure, width](result_array.num_elements())
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            # result_array.store[width=simdwidth](
            #     i, func[dtype, simdwidth](simd_data1, simd_data2)
            # )
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[closure, width](array1.num_elements())
        return result_array

    # TODO: add this function for other backends
    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[closure, width](array1.num_elements())
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data)
            )

        vectorize[closure, width](array.num_elements())
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)

            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data, intval)
            )

        vectorize[closure, width](array.num_elements())
        return result_array


# This provides a way to bypass bitpacking issues with Bool
fn bool_simd_store[
    width: Int
](ptr: DTypePointer[DType.bool], start: Int, val: SIMD[DType.bool, width]):
    """
    Work around function for storing bools from a simd into a DTypePointer.

    Parameters:
        width: Number of items to be retrieved.

    Args:
        ptr: Pointer to be retreived from.
        start: Start position in pointer.
        val: Value to store at locations.
    """
    (ptr + start).simd_strided_store[width=width, T=Int](val, 1)


struct VectorizedUnroll[unroll_factor: Int = 1](Backend):
    """
    Vectorized Backend Struct.
    Parameters
        unroll_factor: factor by which loops are unrolled.

    Defualt Numojo computation backend takes advantage of SIMD.
    Uses defualt simdwidth.
    """

    fn __init__(inout self: Self):
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
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            var simd_data3 = array3.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)

            result_array.store[width=simdwidth](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array.num_elements()
        )

        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data1, simd_data2)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array.load[width=simdwidth](i)
            var simd_data2 = scalar
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data1, simd_data2)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array.num_elements()
        )
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            # result_array.store[width=simdwidth](
            #     i, func[dtype, simdwidth](simd_data1, simd_data2)
            # )
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)
            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array.num_elements()
        )
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)

            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data, intval)
            )

        vectorize[closure, width, unroll_factor=unroll_factor](
            array.num_elements()
        )
        return result_array


struct Parallelized(Backend):
    """
    Parrallelized Backend Struct.

    Currently an order of magnitude slower than Vectorized for most functions.
    No idea why, Not Reccomened for use at this Time.
    """

    fn __init__(inout self: Self):
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
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data3 = array3.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd_data3),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=simdwidth](i+remainder_offset)
        #     var simd_data2 = array2.load[width=simdwidth](i+remainder_offset)
        #     var simd_data3 = array3.load[width=simdwidth](i+remainder_offset)
        #     result_array.store[width=simdwidth](
        #         i+remainder_offset, SIMD.fma(simd_data1,simd_data2,simd_data3)
        #     )
        # vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )

                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=simdwidth](i+remainder_offset)
        #     var simd_data2 = array2.load[width=simdwidth](i+remainder_offset)
        #     result_array.store[width=simdwidth](
        #         i+remainder_offset, SIMD.fma(simd_data1,simd_data2,simd)
        #     )
        # vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j, func[dtype, simdwidth](simd_data)
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data = array.load[width=simdwidth](i+remainder_offset)
        #     result_array.store[width=simdwidth](
        #         i+remainder_offset, func[dtype, simdwidth](simd_data)
        #     )
        # vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=simdwidth](i+remainder_offset)
        #     var simd_data2 = array2.load[width=simdwidth](i+remainder_offset)
        #     result_array.store[width=simdwidth](
        #         i+remainder_offset, func[dtype, simdwidth](simd_data1, simd_data2)
        #     )
        # vectorize[remainder_closure, width](comps_remainder)
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = scalar
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                # result_array.store[width=simdwidth](
                #     i + comps_per_core * j,
                #     func[dtype, simdwidth](simd_data1, simd_data2),
                # )
                bool_simd_store[simdwidth](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=simdwidth](i+remainder_offset)
        #     var simd_data2 = array2.load[width=simdwidth](i+remainder_offset)
        #     result_array.store[width=simdwidth](
        #         i+remainder_offset, func[dtype, simdwidth](simd_data1, simd_data2)
        #     )
        # vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
                # result_array.store[width=simdwidth](
                #     i + comps_per_core * j,
                #     func[dtype, simdwidth](simd_data1, simd_data2),
                # )
                bool_simd_store[simdwidth](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j, func[dtype, simdwidth](simd_data)
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data = array.load[width=simdwidth](i+remainder_offset)
        #     result_array.store[width=simdwidth](
        #         i+remainder_offset, func[dtype, simdwidth](simd_data)
        #     )
        # vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)

            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data, intval)
            )

        vectorize[closure, width](array.num_elements())
        return result_array


struct VectorizedParallelized(Backend):
    """
    Vectorized and Parrallelized Backend Struct.

    Currently an order of magnitude slower than Vectorized for most functions.
    No idea why, Not Reccomened for use at this Time.
    """

    fn __init__(inout self: Self):
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
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data3 = array3.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd_data3),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            var simd_data3 = array3.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset,
                SIMD.fma(simd_data1, simd_data2, simd_data3),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )

                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j, func[dtype, simdwidth](simd_data)
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset, func[dtype, simdwidth](simd_data)
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array1: A NDArray.
            array2: A NDArray.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = scalar
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = scalar
            result_array.store[width=simdwidth](
                i + remainder_offset,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                # result_array.store[width=simdwidth](
                #     i + comps_per_core * j,
                #     func[dtype, simdwidth](simd_data1, simd_data2),
                # )
                bool_simd_store[simdwidth](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            # result_array.store[width=simdwidth](
            #     i + remainder_offset,
            #     func[dtype, simdwidth](simd_data1, simd_data2),
            # )
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
                bool_simd_store[simdwidth](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias width = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j, func[dtype, simdwidth](simd_data)
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset, func[dtype, simdwidth](simd_data)
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)

            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data, intval)
            )

        vectorize[closure, width](array.num_elements())
        return result_array


struct VectorizedParallelizedNWorkers[num_cores: Int = num_physical_cores()](
    Backend
):
    """
    Vectorized and Parrallelized Backend Struct with manual setting of number of workers.

    Speed ups can be acheived by dividing the work across a number of cores, for Windows
    this number seems to be less than `num_physical_cores()`.
    """

    fn __init__(inout self: Self):
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
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            array3: A NDArray.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        # #var num_cores: Int = num_physical_cores()
        # var simd_ops_per_core: Int = width * (array1.num_elements() // width) // num_cores
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        # var op_count:Int=0
        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data3 = array3.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd_data3),
                )
                # op_count+=1

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            var simd_data3 = array3.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset,
                SIMD.fma(simd_data1, simd_data2, simd_data3),
            )
            # op_count+=1

        # print(op_count)
        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            Both arrays must have the same shape.

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = 1
        # var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )

                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()
        # var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j, func[dtype, simdwidth](simd_data)
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset, func[dtype, simdwidth](simd_data)
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        # var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = scalar
                result_array.store[width=simdwidth](
                    i + comps_per_core * j,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = scalar
            result_array.store[width=simdwidth](
                i + remainder_offset,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()
        # var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=simdwidth](
                    i + comps_per_core * j
                )
                # result_array.store[width=simdwidth](
                #     i + comps_per_core * j,
                #     func[dtype, simdwidth](simd_data1, simd_data2),
                # )
                bool_simd_store[simdwidth](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = array2.load[width=simdwidth](i + remainder_offset)
            # result_array.store[width=simdwidth](
            #     i + remainder_offset,
            #     func[dtype, simdwidth](simd_data1, simd_data2),
            # )
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()
        # var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=simdwidth](
                    i + comps_per_core * j
                )
                var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
                bool_simd_store[simdwidth](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, simdwidth](simd_data1, simd_data2),
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i + remainder_offset)
            var simd_data2 = SIMD[dtype, simdwidth].splat(scalar)
            bool_simd_store[simdwidth](
                result_array.unsafe_ptr(),
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias width = simdwidthof[dtype]()
        # var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=simdwidth](
                    i + comps_per_core * j
                )
                result_array.store[width=simdwidth](
                    i + comps_per_core * j, func[dtype, simdwidth](simd_data)
                )

            vectorize[closure, width](comps_per_core)

        parallelize[par_closure](num_cores, num_cores)

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i + remainder_offset)
            result_array.store[width=simdwidth](
                i + remainder_offset, func[dtype, simdwidth](simd_data)
            )

        vectorize[remainder_closure, width](comps_remainder)
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=simdwidth](i)

            result_array.store[width=simdwidth](
                i, func[dtype, simdwidth](simd_data, intval)
            )

        vectorize[closure, width](array.num_elements())
        return result_array


struct Naive(Backend):
    """
    Naive Backend Struct.

    Just loops for SIMD[Dtype, 1] equations
    """

    fn __init__(inout self: Self):
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
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)
            var simd_data3 = array3.load[width=1](i)
            result_array.store[width=1](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)

            result_array.store[width=1](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())

        for i in range(array.num_elements()):
            var simd_data = func[dtype, 1](array.load[width=1](i))
            result_array.store[width=1](i, simd_data)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)
            result_array.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())

        for i in range(array.num_elements()):
            var simd_data1 = array.load[width=1](i)
            var simd_data2 = scalar
            result_array.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)
            # result_array.store[width=1](
            #     i, func[dtype, 1](simd_data1, simd_data2)
            # )
            bool_simd_store[1](
                result_array.unsafe_ptr(),
                i,
                func[dtype, 1](simd_data1, simd_data2),
            )
        return result_array

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: SIMD[dtype, 1]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = scalar
            bool_simd_store[1](
                result_array.unsafe_ptr(),
                i,
                func[dtype, 1](simd_data1, simd_data2),
            )
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )

        for i in range(array.num_elements()):
            var simd_data = func[dtype, 1](array.load[width=1](i))
            result_array.store[width=1](i, simd_data)
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())

        for i in range(array.num_elements()):
            var simd_data1 = array.load[width=1](i)
            result_array.store[width=1](i, func[dtype, 1](simd_data1, intval))
        return result_array


struct VectorizedVerbose(Backend):
    """
    Vectorized Backend Struct.

    Defualt Numojo computation backend takes advantage of SIMD.
    Uses defualt simdwidth.
    """

    fn __init__(inout self: Self):
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
        """

        if (
            array1.shape() != array2.shape()
            and array1.shape() != array3.shape()
        ):
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array1.num_elements() // width), width):
            var simd_data1 = array1.load[width=width](i)
            var simd_data2 = array2.load[width=width](i)
            var simd_data3 = array3.load[width=width](i)
            result_array.store[width=width](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )

        if array1.num_elements() % width != 0:
            for i in range(
                width * (array1.num_elements() // width),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)
                var simd_data3 = array3.load[width=1](i)
                result_array.store[width=1](
                    i, SIMD.fma(simd_data1, simd_data2, simd_data3)
                )
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array1.num_elements() // width), width):
            var simd_data1 = array1.load[width=width](i)
            var simd_data2 = array2.load[width=width](i)

            result_array.store[width=width](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )

        if array1.num_elements() % width != 0:
            for i in range(
                width * (array1.num_elements() // width),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)

                result_array.store[width=1](
                    i, SIMD.fma(simd_data1, simd_data2, simd)
                )
        return result_array

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
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array.num_elements() // width), width):
            var simd_data = array.load[width=width](i)
            result_array.store[width=width](i, func[dtype, width](simd_data))

        if array.num_elements() % width != 0:
            for i in range(
                width * (array.num_elements() // width),
                array.num_elements(),
            ):
                var simd_data = func[dtype, 1](array.load[width=1](i))
                result_array.store[width=1](i, simd_data)
        return result_array

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
            A a new NDArray that is NDArray with the function func applied.
        """

        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array1.num_elements() // width), width):
            var simd_data1 = array1.load[width=width](i)
            var simd_data2 = array2.load[width=width](i)
            result_array.store[width=width](
                i, func[dtype, width](simd_data1, simd_data2)
            )

        if array1.num_elements() % width != 0:
            for i in range(
                width * (array1.num_elements() // width),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)
                result_array.store[width=1](
                    i, func[dtype, 1](simd_data1, simd_data2)
                )
        return result_array

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

        Parameters:
            dtype: The element type.
            func: The SIMD function to to apply.

        Args:
            array: A NDArray.
            scalar: A Scalars.

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array.num_elements() // width), width):
            var simd_data1 = array.load[width=width](i)
            var simd_data2 = scalar
            result_array.store[width=width](
                i, func[dtype, width](simd_data1, simd_data2)
            )

        if array.num_elements() % width != 0:
            for i in range(
                width * (array.num_elements() // width),
                array.num_elements(),
            ):
                var simd_data1 = array.load[width=1](i)
                var simd_data2 = scalar
                result_array.store[width=1](
                    i, func[dtype, 1](simd_data1, simd_data2)
                )
        return result_array

    fn math_func_compare_2_arrays[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[DType.bool]:
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array1.num_elements() // width), width):
            var simd_data1 = array1.load[width=width](i)
            var simd_data2 = array2.load[width=width](i)
            # result_array.store[width=simdwidth](
            #     i, func[dtype, width](simd_data1, simd_data2)
            # )
            bool_simd_store[width](
                result_array.unsafe_ptr(),
                i,
                func[dtype, width](simd_data1, simd_data2),
            )
        if array1.num_elements() % width != 0:
            for i in range(
                width * (array1.num_elements() // width),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)
                # result_array.store[width=1](
                #     i, func[dtype, 1](simd_data1, simd_data2)
                # )
                bool_simd_store[1](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, 1](simd_data1, simd_data2),
                )
        return result_array

    fn math_func_compare_array_and_scalar[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        self: Self, array1: NDArray[dtype], scalar: Scalar[dtype]
    ) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape()
        )
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array1.num_elements() // width), width):
            var simd_data1 = array1.load[width=width](i)
            var simd_data2 = SIMD[dtype, width].splat(scalar)
            bool_simd_store[width](
                result_array.unsafe_ptr(),
                i,
                func[dtype, width](simd_data1, simd_data2),
            )
        if array1.num_elements() % width != 0:
            for i in range(
                width * (array1.num_elements() // width),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = SIMD[dtype, 1].splat(scalar)
                bool_simd_store[1](
                    result_array.unsafe_ptr(),
                    i,
                    func[dtype, 1](simd_data1, simd_data2),
                )
        return result_array

    fn math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) raises -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array.num_elements() // width), width):
            var simd_data = array.load[width=width](i)
            result_array.store[width=width](i, func[dtype, width](simd_data))

        if array.num_elements() % width != 0:
            for i in range(
                width * (array.num_elements() // width),
                array.num_elements(),
            ):
                var simd_data = func[dtype, 1](array.load[width=1](i))
                result_array.store[width=1](i, simd_data)
        return result_array

    fn math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array1: NDArray[dtype], intval: Int) raises -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias width = simdwidthof[dtype]()
        for i in range(0, width * (array1.num_elements() // width), width):
            var simd_data1 = array1.load[width=width](i)

            result_array.store[width=width](
                i, func[dtype, width](simd_data1, intval)
            )

        if array1.num_elements() % width != 0:
            for i in range(
                width * (array1.num_elements() // width),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                result_array.store[width=1](
                    i, func[dtype, 1](simd_data1, intval)
                )
        return result_array
