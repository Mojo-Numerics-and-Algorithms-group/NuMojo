# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator unary op kernels
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Unary Ops Kernels (numojo.core.accelerator.kernels.unary_ops)
==============================================================

GPU kernels for unary operations.

GPU kernel functions and launch helpers for element-wise unary operations
on contiguous AcceleratorNDArray buffers.

Exports
-------
- Unary operation kernel functions for GPU acceleration.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)

# ===----------------------------------------------------------------------===#
# External
# ===----------------------------------------------------------------------===#
from max.gpu.host import DeviceContext

# ===----------------------------------------------------------------------===#
# Local
# ===----------------------------------------------------------------------===#
from .binary_ops import launch_config


def neg_kernel[
    dtype: DType
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    """GPU kernel: `result[i] = -a[i]` for contiguous buffers."""
    var i = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if i < size:
        result[unsafe_offset=i] = -a[unsafe_offset=i]


def launch_neg[
    dtype: DType
](
    context: DeviceContext,
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
    sync: Bool = True,
) raises:
    """Launch the GPU negation kernel over `size` contiguous elements."""
    var grid_dim: Int
    var block_dim: Int
    grid_dim, block_dim = launch_config(size)
    context.enqueue_function[neg_kernel[dtype]](
        result,
        a,
        size,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )
    if sync:
        context.synchronize()
