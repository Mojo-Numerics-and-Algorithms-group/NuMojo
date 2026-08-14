# ===----------------------------------------------------------------------=== #
# NuMojo: Math backend
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Math operations backend (numojo.routines.operations.backend).
----------------------------------------------------------------
Defines vectorized backend structures and reusable SIMD math primitives consumed by the math submodules.
"""

from std.algorithm.functional import vectorize
from max.algorithm import parallelize
from std.sys import simd_width_of
from std.sys.info import num_performance_cores
from std.builtin.simd import FastMathFlag

from numojo.core.ndarray import NDArray
from numojo.routines.creation import _0darray
from numojo.routines.manipulation import broadcast_to

comptime MIN_SIMD_WIDTHS_PER_TASK = 8
"""Minimum number of SIMD-widths of work each parallel task should get before
splitting across cores is worth the thread-dispatch overhead. This is a heuristic
that can be tuned.
"""


@always_inline
def _num_tasks_for(size: Int, simd_width: Int) -> Int:
    """
    Decide how many parallel tasks to split `size` elements into.

    Returns 1 (i.e. no parallelism) when there isn't enough work per task
    to justify the overhead of spawning threads.
    """
    var min_chunk = simd_width * MIN_SIMD_WIDTHS_PER_TASK
    var max_tasks_by_size = size // min_chunk
    if max_tasks_by_size <= 1:
        return 1
    var cores = num_performance_cores()
    return min(cores, max_tasks_by_size)


# NOTE: These chunk-application helpers are top-level (non-nested) functions
# on purpose. This is to prevent the mojo compiler crash from nested clsoure funcs.
@always_inline
def _apply_unary_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](SIMD[type, simd_w]) capturing -> SIMD[
        type, simd_w
    ],
    src_origin: Origin,
    dst_origin: MutOrigin,
](
    src: Pointer[Scalar[dtype], src_origin],
    dst: Pointer[Scalar[dtype], dst_origin],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        dst.unsafe_store(
            i, kernel[dtype, width](src.unsafe_load[width=width](i))
        )
        i += width
    while i < end:
        dst.unsafe_store(i, kernel[dtype, 1](src.unsafe_load[width=1](i)))
        i += 1


@always_inline
def _apply_binary_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) capturing -> SIMD[type, simd_w],
    src1_origin: Origin,
    src2_origin: Origin,
    dst_origin: MutOrigin,
](
    src1: Pointer[Scalar[dtype], src1_origin],
    src2: Pointer[Scalar[dtype], src2_origin],
    dst: Pointer[Scalar[dtype], dst_origin],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        dst.unsafe_store(
            i,
            kernel[dtype, width](
                src1.unsafe_load[width=width](i),
                src2.unsafe_load[width=width](i),
            ),
        )
        i += width
    while i < end:
        dst.unsafe_store(
            i,
            kernel[dtype, 1](
                src1.unsafe_load[width=1](i), src2.unsafe_load[width=1](i)
            ),
        )
        i += 1


@always_inline
def _apply_binary_scalar_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) capturing -> SIMD[type, simd_w],
    *,
    scalar_first: Bool,
    src_origin: Origin,
    dst_origin: MutOrigin,
](
    src: Pointer[Scalar[dtype], src_origin],
    dst: Pointer[Scalar[dtype], dst_origin],
    scalar: Scalar[dtype],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        var data = src.unsafe_load[width=width](i)
        comptime if scalar_first:
            dst.unsafe_store(i, kernel[dtype, width](scalar, data))
        else:
            dst.unsafe_store(i, kernel[dtype, width](data, scalar))
        i += width
    while i < end:
        var data = src.unsafe_load[width=1](i)
        comptime if scalar_first:
            dst.unsafe_store(i, kernel[dtype, 1](scalar, data))
        else:
            dst.unsafe_store(i, kernel[dtype, 1](data, scalar))
        i += 1


@always_inline
def _apply_binary_int_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], Int
    ) capturing -> SIMD[type, simd_w],
    src_origin: Origin,
    dst_origin: MutOrigin,
](
    src: Pointer[Scalar[dtype], src_origin],
    dst: Pointer[Scalar[dtype], dst_origin],
    intval: Int,
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        dst.unsafe_store(
            i, kernel[dtype, width](src.unsafe_load[width=width](i), intval)
        )
        i += width
    while i < end:
        dst.unsafe_store(
            i, kernel[dtype, 1](src.unsafe_load[width=1](i), intval)
        )
        i += 1


@always_inline
def _apply_binary_predicate_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) capturing -> SIMD[DType.bool, simd_w],
    src1_origin: Origin,
    src2_origin: Origin,
    dst_origin: MutOrigin,
](
    src1: Pointer[Scalar[dtype], src1_origin],
    src2: Pointer[Scalar[dtype], src2_origin],
    dst: Pointer[Scalar[DType.bool], dst_origin],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        bool_simd_store[width](
            dst,
            i,
            kernel[dtype, width](
                src1.unsafe_load[width=width](i),
                src2.unsafe_load[width=width](i),
            ),
        )
        i += width
    while i < end:
        bool_simd_store[1](
            dst,
            i,
            kernel[dtype, 1](
                src1.unsafe_load[width=1](i), src2.unsafe_load[width=1](i)
            ),
        )
        i += 1


@always_inline
def _apply_binary_predicate_scalar_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) capturing -> SIMD[DType.bool, simd_w],
    src_origin: Origin,
    dst_origin: MutOrigin,
](
    src: Pointer[Scalar[dtype], src_origin],
    dst: Pointer[Scalar[DType.bool], dst_origin],
    scalar: Scalar[dtype],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        bool_simd_store[width](
            dst,
            i,
            kernel[dtype, width](
                src.unsafe_load[width=width](i), SIMD[dtype, width](scalar)
            ),
        )
        i += width
    while i < end:
        bool_simd_store[1](
            dst,
            i,
            kernel[dtype, 1](
                src.unsafe_load[width=1](i), SIMD[dtype, 1](scalar)
            ),
        )
        i += 1


@always_inline
def _apply_unary_predicate_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](SIMD[type, simd_w]) capturing -> SIMD[
        DType.bool, simd_w
    ],
    src_origin: Origin,
    dst_origin: MutOrigin,
](
    src: Pointer[Scalar[dtype], src_origin],
    dst: Pointer[Scalar[DType.bool], dst_origin],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        bool_simd_store[width](
            dst, i, kernel[dtype, width](src.unsafe_load[width=width](i))
        )
        i += width
    while i < end:
        bool_simd_store[1](
            dst, i, kernel[dtype, 1](src.unsafe_load[width=1](i))
        )
        i += 1


@always_inline
def _apply_ternary_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], SIMD[type, simd_w], SIMD[type, simd_w]
    ) capturing -> SIMD[type, simd_w],
    src1_origin: Origin,
    src2_origin: Origin,
    src3_origin: Origin,
    dst_origin: MutOrigin,
](
    src1: Pointer[Scalar[dtype], src1_origin],
    src2: Pointer[Scalar[dtype], src2_origin],
    src3: Pointer[Scalar[dtype], src3_origin],
    dst: Pointer[Scalar[dtype], dst_origin],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        dst.unsafe_store(
            i,
            kernel(
                src1.unsafe_load[width=width](i),
                src2.unsafe_load[width=width](i),
                src3.unsafe_load[width=width](i),
            ),
        )
        i += width
    while i < end:
        dst.unsafe_store(
            i,
            kernel[dtype, 1](
                src1.unsafe_load[width=1](i),
                src2.unsafe_load[width=1](i),
                src3.unsafe_load[width=1](i),
            ),
        )
        i += 1


@always_inline
def _apply_ternary_scalar_chunk[
    dtype: DType,
    width: Int,
    kernel: def[type: DType, simd_w: Int](
        SIMD[type, simd_w], SIMD[type, simd_w], SIMD[type, simd_w]
    ) capturing -> SIMD[type, simd_w],
    src1_origin: Origin,
    src2_origin: Origin,
    dst_origin: MutOrigin,
](
    src1: Pointer[Scalar[dtype], src1_origin],
    src2: Pointer[Scalar[dtype], src2_origin],
    dst: Pointer[Scalar[dtype], dst_origin],
    scalar: Scalar[dtype],
    start: Int,
    end: Int,
):
    var i = start
    while i + width <= end:
        dst.unsafe_store(
            i,
            kernel(
                src1.unsafe_load[width=width](i),
                src2.unsafe_load[width=width](i),
                SIMD[dtype, width](scalar),
            ),
        )
        i += width
    while i < end:
        dst.unsafe_store(
            i,
            kernel[dtype, 1](
                src1.unsafe_load[width=1](i),
                src2.unsafe_load[width=1](i),
                SIMD[dtype, 1](scalar),
            ),
        )
        i += 1


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

    def __init__(out self):
        pass

    @staticmethod
    def apply_unary[
        dtype: DType,
        simd_width: Int,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
    def apply_binary[
        dtype: DType,
        simd_width: Int,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
    def apply_unary_predicate[
        dtype: DType,
        simd_width: Int,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w]
        ) capturing -> SIMD[DType.bool, simd_w],
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
    def apply_binary_predicate[
        dtype: DType,
        simd_width: Int,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[DType.bool, simd_w],
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
    def apply_unary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
                val=kernel[dtype, 1](
                    (array._buf.ptr.unsafe_offset(array.offset))[]
                )
            )
            return result_array^

        var result_array: NDArray[dtype] = NDArray[dtype](array.shape)
        comptime width = simd_width_of[dtype]()
        var num_tasks = _num_tasks_for(array.size, width)
        var src = array.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        if num_tasks == 1:
            _apply_unary_chunk[dtype, width, kernel](src, dst, 0, array.size)
        else:
            var chunk_size = (array.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array.size)
                if end > start:
                    _apply_unary_chunk[dtype, width, kernel](
                        src, dst, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_binary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
            var common_shape = array1.shape.broadcast(array2.shape)
            return Self.apply_binary[dtype, kernel](
                broadcast_to(array1, common_shape),
                broadcast_to(array2, common_shape),
            )

        var result_array: NDArray[dtype] = NDArray[dtype](array1.shape)
        comptime width = simd_width_of[dtype]()
        var src1 = array1.unsafe_ptr()
        var src2 = array2.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(result_array.size, width)
        if num_tasks == 1:
            _apply_binary_chunk[dtype, width, kernel](
                src1, src2, dst, 0, result_array.size
            )
        else:
            var chunk_size = (result_array.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, result_array.size)
                if end > start:
                    _apply_binary_chunk[dtype, width, kernel](
                        src1, src2, dst, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_binary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
        var src = array.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(result_array.size, width)
        if num_tasks == 1:
            _apply_binary_scalar_chunk[
                dtype, width, kernel, scalar_first=False
            ](src, dst, scalar, 0, result_array.size)
        else:
            var chunk_size = (result_array.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, result_array.size)
                if end > start:
                    _apply_binary_scalar_chunk[
                        dtype, width, kernel, scalar_first=False
                    ](src, dst, scalar, start, end)

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_binary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
        var src = array.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(result_array.size, width)
        if num_tasks == 1:
            _apply_binary_scalar_chunk[dtype, width, kernel, scalar_first=True](
                src, dst, scalar, 0, result_array.size
            )
        else:
            var chunk_size = (result_array.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, result_array.size)
                if end > start:
                    _apply_binary_scalar_chunk[
                        dtype, width, kernel, scalar_first=True
                    ](src, dst, scalar, start, end)

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_binary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], Int
        ) capturing -> SIMD[type, simd_w],
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
        var src = array.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(array.size, width)
        if num_tasks == 1:
            _apply_binary_int_chunk[dtype, width, kernel](
                src, dst, intval, 0, array.size
            )
        else:
            var chunk_size = (array.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array.size)
                if end > start:
                    _apply_binary_int_chunk[dtype, width, kernel](
                        src, dst, intval, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_binary_predicate[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[DType.bool, simd_w],
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
            # Shapes differ: broadcast both operands (zero-copy views) to
            # their common shape, then fall through to the equal-shape path.
            var common_shape = array1.shape.broadcast(array2.shape)
            return Self.apply_binary_predicate[dtype, kernel](
                broadcast_to(array1, common_shape),
                broadcast_to(array2, common_shape),
            )

        var result_array: NDArray[DType.bool] = NDArray[DType.bool](
            array1.shape
        )
        comptime width = simd_width_of[DType.bool]()
        var src1 = array1.unsafe_ptr()
        var src2 = array2.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(array1.size, width)
        if num_tasks == 1:
            _apply_binary_predicate_chunk[dtype, width, kernel](
                src1, src2, dst, 0, array1.size
            )
        else:
            var chunk_size = (array1.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array1.size)
                if end > start:
                    _apply_binary_predicate_chunk[dtype, width, kernel](
                        src1, src2, dst, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_binary_predicate[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[DType.bool, simd_w],
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
        var src = array1.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(array1.size, width)
        if num_tasks == 1:
            _apply_binary_predicate_scalar_chunk[dtype, width, kernel](
                src, dst, scalar, 0, array1.size
            )
        else:
            var chunk_size = (array1.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array1.size)
                if end > start:
                    _apply_binary_predicate_scalar_chunk[dtype, width, kernel](
                        src, dst, scalar, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_unary_predicate[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w]
        ) capturing -> SIMD[DType.bool, simd_w],
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
        var src = array.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(array.size, width)
        if num_tasks == 1:
            _apply_unary_predicate_chunk[dtype, width, kernel](
                src, dst, 0, array.size
            )
        else:
            var chunk_size = (array.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array.size)
                if end > start:
                    _apply_unary_predicate_chunk[dtype, width, kernel](
                        src, dst, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_ternary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
        var src1 = array1.unsafe_ptr()
        var src2 = array2.unsafe_ptr()
        var src3 = array3.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(array1.size, width)
        if num_tasks == 1:
            _apply_ternary_chunk[dtype, width, kernel](
                src1, src2, src3, dst, 0, array1.size
            )
        else:
            var chunk_size = (array1.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array1.size)
                if end > start:
                    _apply_ternary_chunk[dtype, width, kernel](
                        src1, src2, src3, dst, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^

    @staticmethod
    def apply_ternary[
        dtype: DType,
        kernel: def[type: DType, simd_w: Int](
            SIMD[type, simd_w], SIMD[type, simd_w], SIMD[type, simd_w]
        ) capturing -> SIMD[type, simd_w],
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
        var src1 = array1.unsafe_ptr()
        var src2 = array2.unsafe_ptr()
        var dst = result_array.unsafe_ptr()

        var num_tasks = _num_tasks_for(array1.size, width)
        if num_tasks == 1:
            _apply_ternary_scalar_chunk[dtype, width, kernel](
                src1, src2, dst, scalar, 0, array1.size
            )
        else:
            var chunk_size = (array1.size + num_tasks - 1) // num_tasks

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, array1.size)
                if end > start:
                    _apply_ternary_scalar_chunk[dtype, width, kernel](
                        src1, src2, dst, scalar, start, end
                    )

            parallelize[worker](num_tasks)

        return result_array^


# This provides a way to bypass bitpacking issues with Bool
def bool_simd_store[
    ptr_origin: MutOrigin,
    //,
    simd_width: Int,
](
    ptr: Pointer[Scalar[DType.bool], ptr_origin],
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
    (ptr.unsafe_offset(start)).unsafe_strided_store[width=simd_width](
        val=val, stride=1
    )
