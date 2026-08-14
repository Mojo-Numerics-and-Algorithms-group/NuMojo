# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator binary op kernels
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Binary op GPU kernels (numojo.core.accelerator.kernels.binary_ops)
-----------------------------------------------------------------
GPU kernel functions and launch helpers for elementwise binary operations
on contiguous `AcceleratorNDArray` buffers.
"""
# ===----------------------------------------------------------------------===#
# Stdlib
# ===----------------------------------------------------------------------===#
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)

# ===----------------------------------------------------------------------===#
# External
# ===----------------------------------------------------------------------===#
from max.gpu.host import DeviceContext


comptime ADD = 0
comptime SUB = 1
comptime MUL = 2
comptime DIV = 3


@always_inline
def _binary_op[
    dtype: DType, op_code: Int
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    comptime if op_code == ADD:
        return a + b
    elif op_code == SUB:
        return a - b
    elif op_code == MUL:
        return a * b
    else:
        return a / b


def binary_op_kernel[
    dtype: DType, op_code: Int
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    """GPU kernel: `result[i] = op(a[i], b[i])` for contiguous buffers."""
    var i = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if i < size:
        result[unsafe_offset=i] = _binary_op[dtype, op_code](
            a[unsafe_offset=i], b[unsafe_offset=i]
        )


def launch_config(size: Int) -> Tuple[Int, Int]:
    """Compute (grid_dim, block_dim) for a one-thread-per-element launch.

    Returns:
        A tuple of (number of blocks, threads per block).
    """
    comptime threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    return (max(1, num_blocks), threads_per_block)


def launch_binary_op[
    dtype: DType, op_code: Int
](
    context: DeviceContext,
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
    sync: Bool = True,
) raises:
    """Launch the GPU binary-op kernel over `size` contiguous elements."""
    var grid_dim: Int
    var block_dim: Int
    grid_dim, block_dim = launch_config(size)
    context.enqueue_function[binary_op_kernel[dtype, op_code]](
        result,
        a,
        b,
        size,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )
    if sync:
        context.synchronize()
