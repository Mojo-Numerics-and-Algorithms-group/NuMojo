"""
# ===----------------------------------------------------------------------=== #
# Implements generic reusable functions for math
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

from testing import assert_raises
from algorithm.functional import parallelize, vectorize, num_physical_cores

from ..traits.backend import Backend
from .ndarray import NDArray, NDArrayShape


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

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            array3: A NDArray

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
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            var simd_data3 = array3.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )

        vectorize[closure, opt_nelts](array1.num_elements())
        return result_array

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        simd: SIMD[dtype, 1],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            simd: A SIMD[dtype,1] value to be added

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
            # var simd_data3 = array3.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[closure, opt_nelts](array1.num_elements())
        return result_array

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

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

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

    fn _math_func_compare_2_tensors[
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
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        vectorize[closure, opt_nelts](array1.num_elements())
        return result_array

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        vectorize[closure, opt_nelts](array.num_elements())
        return result_array

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)

            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data, intval)
            )

        vectorize[closure, opt_nelts](array.num_elements())
        return result_array


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

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            array3: A NDArray

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
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            var simd_data3 = array3.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

    fn _math_func_fma[
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
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added

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
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

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
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
            array.num_elements()
        )

        return result_array

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

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

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

    fn _math_func_compare_2_tensors[
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
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
            array1.num_elements()
        )
        return result_array

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
            array.num_elements()
        )
        return result_array

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)

            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data, intval)
            )

        vectorize[closure, opt_nelts, unroll_factor=unroll_factor](
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

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            array3: A NDArray

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
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data3 = array3.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd_data3),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=opt_nelts](i+remainder_offset)
        #     var simd_data2 = array2.load[width=opt_nelts](i+remainder_offset)
        #     var simd_data3 = array3.load[width=opt_nelts](i+remainder_offset)
        #     result_array.store[width=opt_nelts](
        #         i+remainder_offset, SIMD.fma(simd_data1,simd_data2,simd_data3)
        #     )
        # vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_fma[
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
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )

                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=opt_nelts](i+remainder_offset)
        #     var simd_data2 = array2.load[width=opt_nelts](i+remainder_offset)
        #     result_array.store[width=opt_nelts](
        #         i+remainder_offset, SIMD.fma(simd_data1,simd_data2,simd)
        #     )
        # vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

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
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j, func[dtype, opt_nelts](simd_data)
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data = array.load[width=opt_nelts](i+remainder_offset)
        #     result_array.store[width=opt_nelts](
        #         i+remainder_offset, func[dtype, opt_nelts](simd_data)
        #     )
        # vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

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
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    func[dtype, opt_nelts](simd_data1, simd_data2),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=opt_nelts](i+remainder_offset)
        #     var simd_data2 = array2.load[width=opt_nelts](i+remainder_offset)
        #     result_array.store[width=opt_nelts](
        #         i+remainder_offset, func[dtype, opt_nelts](simd_data1, simd_data2)
        #     )
        # vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_compare_2_tensors[
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
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    func[dtype, opt_nelts](simd_data1, simd_data2),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data1 = array1.load[width=opt_nelts](i+remainder_offset)
        #     var simd_data2 = array2.load[width=opt_nelts](i+remainder_offset)
        #     result_array.store[width=opt_nelts](
        #         i+remainder_offset, func[dtype, opt_nelts](simd_data1, simd_data2)
        #     )
        # vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j, func[dtype, opt_nelts](simd_data)
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()
        # @parameter
        # fn remainder_closure[simdwidth: Int](i: Int):
        #     var simd_data = array.load[width=opt_nelts](i+remainder_offset)
        #     result_array.store[width=opt_nelts](
        #         i+remainder_offset, func[dtype, opt_nelts](simd_data)
        #     )
        # vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)

            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data, intval)
            )

        vectorize[closure, opt_nelts](array.num_elements())
        return result_array


struct VectorizedParallelized(Backend):
    """
    Vectorized and Parrallelized Backend Struct.

    Currently an order of magnitude slower than Vectorized for most functions.
    No idea why, Not Reccomened for use at this Time.
    """

    fn __init__(inout self: Self):
        pass

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            array3: A NDArray

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
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data3 = array3.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd_data3),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i + remainder_offset)
            var simd_data2 = array2.load[width=opt_nelts](i + remainder_offset)
            var simd_data3 = array3.load[width=opt_nelts](i + remainder_offset)
            result_array.store[width=opt_nelts](
                i + remainder_offset,
                SIMD.fma(simd_data1, simd_data2, simd_data3),
            )

        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_fma[
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
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias opt_nelts = 1
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )

                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    SIMD.fma(simd_data1, simd_data2, simd),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i + remainder_offset)
            var simd_data2 = array2.load[width=opt_nelts](i + remainder_offset)
            result_array.store[width=opt_nelts](
                i + remainder_offset, SIMD.fma(simd_data1, simd_data2, simd)
            )

        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

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
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j, func[dtype, opt_nelts](simd_data)
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i + remainder_offset)
            result_array.store[width=opt_nelts](
                i + remainder_offset, func[dtype, opt_nelts](simd_data)
            )

        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

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
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    func[dtype, opt_nelts](simd_data1, simd_data2),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i + remainder_offset)
            var simd_data2 = array2.load[width=opt_nelts](i + remainder_offset)
            result_array.store[width=opt_nelts](
                i + remainder_offset,
                func[dtype, opt_nelts](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_compare_2_tensors[
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
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array1.num_elements() // num_cores
        var comps_remainder: Int = array1.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = array1.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                var simd_data2 = array2.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j,
                    func[dtype, opt_nelts](simd_data1, simd_data2),
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=opt_nelts](i + remainder_offset)
            var simd_data2 = array2.load[width=opt_nelts](i + remainder_offset)
            result_array.store[width=opt_nelts](
                i + remainder_offset,
                func[dtype, opt_nelts](simd_data1, simd_data2),
            )

        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = array.num_elements() // num_cores
        var comps_remainder: Int = array.num_elements() % num_cores
        var remainder_offset: Int = num_cores * comps_per_core

        @parameter
        fn par_closure(j: Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = array.load[width=opt_nelts](
                    i + comps_per_core * j
                )
                result_array.store[width=opt_nelts](
                    i + comps_per_core * j, func[dtype, opt_nelts](simd_data)
                )

            vectorize[closure, opt_nelts](comps_per_core)

        parallelize[par_closure]()

        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i + remainder_offset)
            result_array.store[width=opt_nelts](
                i + remainder_offset, func[dtype, opt_nelts](simd_data)
            )

        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_array

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()

        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = array.load[width=opt_nelts](i)

            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data, intval)
            )

        vectorize[closure, opt_nelts](array.num_elements())
        return result_array


struct Naive(Backend):
    """
    Naive Backend Struct.

    Just loops for SIMD[Dtype, 1] equations
    """

    fn __init__(inout self: Self):
        pass

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            array3: A NDArray

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
        alias opt_nelts = simdwidthof[dtype]()

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)
            var simd_data3 = array3.load[width=1](i)
            result_array.store[width=1](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )
        return result_array

    fn _math_func_fma[
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
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias opt_nelts = simdwidthof[dtype]()

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)

            result_array.store[width=1](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )
        return result_array

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
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())

        for i in range(array.num_elements()):
            var simd_data = func[dtype, 1](array.load[width=1](i))
            result_array.store[width=1](i, simd_data)
        return result_array

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

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

        for i in range(array1.num_elements()):
            var simd_data1 = array1.load[width=1](i)
            var simd_data2 = array2.load[width=1](i)
            result_array.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
        return result_array

    fn _math_func_compare_2_tensors[
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
            result_array.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
        return result_array

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )

        for i in range(array.num_elements()):
            var simd_data = func[dtype, 1](array.load[width=1](i))
            result_array.store[width=1](i, simd_data)
        return result_array

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array: NDArray[dtype], intval: Int) -> NDArray[dtype]:
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

    fn _math_func_fma[
        dtype: DType,
    ](
        self: Self,
        array1: NDArray[dtype],
        array2: NDArray[dtype],
        array3: NDArray[dtype],
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray
            array2: A NDArray
            array3: A NDArray

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
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (array1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            var simd_data3 = array3.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, SIMD.fma(simd_data1, simd_data2, simd_data3)
            )

        if array1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array1.num_elements() // opt_nelts),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)
                var simd_data3 = array3.load[width=1](i)
                result_array.store[width=1](
                    i, SIMD.fma(simd_data1, simd_data2, simd_data3)
                )
        return result_array

    fn _math_func_fma[
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
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.

        Args:
            array1: A NDArray.
            array2: A NDArray.
            simd: A SIMD[dtype,1] value to be added

        Returns:
            A a new NDArray that is NDArray with the function func applied.
        """
        if array1.shape() != array2.shape():
            raise Error(
                "Shape Mismatch error shapes must match for this function"
            )
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (array1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)

            result_array.store[width=opt_nelts](
                i, SIMD.fma(simd_data1, simd_data2, simd)
            )

        if array1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array1.num_elements() // opt_nelts),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)

                result_array.store[width=1](
                    i, SIMD.fma(simd_data1, simd_data2, simd)
                )
        return result_array

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
        var result_array: NDArray[dtype] = NDArray[dtype](array.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (array.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data = array.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        if array.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array.num_elements() // opt_nelts),
                array.num_elements(),
            ):
                var simd_data = func[dtype, 1](array.load[width=1](i))
                result_array.store[width=1](i, simd_data)
        return result_array

    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](
        self: Self, array1: NDArray[dtype], array2: NDArray[dtype]
    ) raises -> NDArray[dtype]:
        """
        Apply a SIMD function of two variable and one return to a NDArray

        Constraints:
            Both tensors must have the same shape

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
        for i in range(
            0, opt_nelts * (array1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        if array1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array1.num_elements() // opt_nelts),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)
                result_array.store[width=1](
                    i, func[dtype, 1](simd_data1, simd_data2)
                )
        return result_array

    fn _math_func_compare_2_tensors[
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
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (array1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = array1.load[width=opt_nelts](i)
            var simd_data2 = array2.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        if array1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array1.num_elements() // opt_nelts),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                var simd_data2 = array2.load[width=1](i)
                result_array.store[width=1](
                    i, func[dtype, 1](simd_data1, simd_data2)
                )
        return result_array

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self: Self, array: NDArray[dtype]) -> NDArray[DType.bool]:
        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array.shape()
        )
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (array.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data = array.load[width=opt_nelts](i)
            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        if array.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array.num_elements() // opt_nelts),
                array.num_elements(),
            ):
                var simd_data = func[dtype, 1](array.load[width=1](i))
                result_array.store[width=1](i, simd_data)
        return result_array

    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self: Self, array1: NDArray[dtype], intval: Int) -> NDArray[dtype]:
        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (array1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = array1.load[width=opt_nelts](i)

            result_array.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, intval)
            )

        if array1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (array1.num_elements() // opt_nelts),
                array1.num_elements(),
            ):
                var simd_data1 = array1.load[width=1](i)
                result_array.store[width=1](
                    i, func[dtype, 1](simd_data1, intval)
                )
        return result_array
