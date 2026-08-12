# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator reduction kernels
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Reduction GPU kernels (numojo.core.accelerator.kernels.reduction_ops)
-----------------------------------------------------------------
GPU kernel functions and launch helpers for full-array reduction ops.
"""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace, stack_allocation, alloc

from .binary_ops import launch_config


def sum_reduce_kernel[
    dtype: DType, block_size: Int
](
    partial_sums: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    """GPU kernel: each block reduces its chunk of `a` to one partial sum.

    The host is responsible for summing the `partial_sums` buffer
    (one element per block) into the final scalar result.
    """
    var smem = stack_allocation[
        block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var idx = bid * block_size + tid

    if idx < size:
        smem[tid] = a[idx]
    else:
        smem[tid] = Scalar[dtype](0)
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        partial_sums[bid] = smem[0]


def launch_sum_reduce[
    dtype: DType
](
    context: DeviceContext,
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
) raises -> Scalar[dtype]:
    """Launch the GPU sum-reduction kernel and combine partial sums.

    Args:
        context: The `DeviceContext` backing `a`'s device memory.
        a: Device pointer to the first element to reduce.
        size: Number of contiguous elements to reduce.

    Returns:
        The sum of all `size` elements, as a host scalar.
    """
    comptime threads_per_block = 256
    var num_blocks: Int
    var block_dim_size: Int
    num_blocks, block_dim_size = launch_config(size)

    var partial_sums = context.enqueue_create_buffer[dtype](num_blocks)
    context.enqueue_function[sum_reduce_kernel[dtype, threads_per_block]](
        partial_sums.unsafe_ptr(),
        a,
        size,
        grid_dim=num_blocks,
        block_dim=block_dim_size,
    )

    var host_partial = alloc[Scalar[dtype]](num_blocks)
    partial_sums.enqueue_copy_to(host_partial)
    context.synchronize()

    var result = Scalar[dtype](0)
    for i in range(num_blocks):
        result += host_partial[i]

    host_partial.free()
    return result
