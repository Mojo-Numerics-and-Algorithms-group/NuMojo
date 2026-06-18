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

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


def add_kernel[
    dtype: DType
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    """GPU kernel: `result[i] = a[i] + b[i]` for contiguous buffers."""
    var i = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if i < size:
        result[i] = a[i] + b[i]


def launch_config(size: Int) -> Tuple[Int, Int]:
    """Compute (grid_dim, block_dim) for a one-thread-per-element launch.

    Returns:
        A tuple of (number of blocks, threads per block).
    """
    comptime threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    return (max(1, num_blocks), threads_per_block)


def launch_add[
    dtype: DType
](
    context: DeviceContext,
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
    sync: Bool = True,
) raises:
    """Launch the GPU add kernel over `size` contiguous elements."""
    var grid_dim: Int
    var block_dim: Int
    grid_dim, block_dim = launch_config(size)
    context.enqueue_function[add_kernel[dtype], add_kernel[dtype]](
        result,
        a,
        b,
        size,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )
    if sync:
        context.synchronize()
