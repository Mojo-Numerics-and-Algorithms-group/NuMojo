from gpu import thread_idx, block_dim, block_idx, grid_dim, barrier
from gpu.globals import WARP_SIZE, MAX_THREADS_PER_BLOCK_METADATA
from gpu.warp import sum as warp_sum
from gpu.memory import load, AddressSpace
from memory import UnsafePointer, stack_allocation
from sys import simd_width_of

# temporary defaults.
alias MAX_THREADS_PER_BLOCK = 1024
alias OPTIMAL_BLOCK_SIZE = 256
alias TILE_SIZE = 32
alias VECTOR_WIDTH = 4
alias MAX_WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK // UInt(WARP_SIZE)

fn matrix_add_kernel_vectorized[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: UInt,
    total_threads: UInt,
):
    """Optimized GPU kernel for element-wise matrix addition with vectorization.

    Features:
    - Vectorized memory operations (dtype-specific widths)
    - Coalesced memory access patterns
    - Grid-stride loops
    - Proper bounds checking

    Parameters:
        dtype: Data type of the matrices.

    Args:
        output: Output buffer for result.
        a: First input matrix buffer.
        b: Second input matrix buffer.
        size: Total number of elements.
        total_threads: Total number of threads launched (grid_size * block_size).
    """
    var tid = block_dim.x * block_idx.x + thread_idx.x
    var grid_stride = total_threads

    @parameter
    if dtype == DType.float16:
        # width = 8
        var base_idx = tid * 8
        var idx = base_idx
        while idx + 8 <= size:
            var a_vec = a.load[width=8](idx)
            var b_vec = b.load[width=8](idx)
            var result = a_vec + b_vec
            output.store[width=8](idx, result)
            idx += grid_stride * 8
        while idx < size:
            output[idx] = a[idx] + b[idx]
            idx += grid_stride
    elif dtype == DType.float32:
        # width = 4
        var base_idx = tid * 4
        var idx = base_idx
        while idx + 4 <= size:
            var a_vec = a.load[width=4](idx)
            var b_vec = b.load[width=4](idx)
            var result = a_vec + b_vec
            output.store[width=4](idx, result)
            idx += grid_stride * 4
        while idx < size:
            output[idx] = a[idx] + b[idx]
            idx += grid_stride
    elif dtype == DType.float64:
        # width = 2
        var base_idx = tid * 2
        var idx = base_idx
        while idx + 2 <= size:
            var a_vec = a.load[width=2](idx)
            var b_vec = b.load[width=2](idx)
            var result = a_vec + b_vec
            output.store[width=2](idx, result)
            idx += grid_stride * 2
        while idx < size:
            output[idx] = a[idx] + b[idx]
            idx += grid_stride
    else:
        # Scalar fallback with grid-stride
        var idx = tid
        while idx < size:
            output[idx] = a[idx] + b[idx]
            idx += grid_stride

fn matrix_sub_kernel_vectorized[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: UInt,
    total_threads: UInt,
):
    """Optimized GPU kernel for element-wise matrix subtraction with vectorization.

    Features:
    - Vectorized memory operations (dtype-specific widths)
    - Coalesced memory access patterns
    - Grid-stride loops
    - Proper bounds checking

    Parameters:
        dtype: Data type of the matrices.

    Args:
        output: Output buffer for result.
        a: First input matrix buffer.
        b: Second input matrix buffer.
        size: Total number of elements.
        total_threads: Total number of threads launched (grid_size * block_size).
    """
    var tid = block_dim.x * block_idx.x + thread_idx.x
    var grid_stride = total_threads

    @parameter
    if dtype == DType.float16:
        # width = 8
        var base_idx = tid * 8
        var idx = base_idx
        while idx + 8 <= size:
            var a_vec = a.load[width=8](idx)
            var b_vec = b.load[width=8](idx)
            var result = a_vec - b_vec
            output.store[width=8](idx, result)
            idx += grid_stride * 8
        while idx < size:
            output[idx] = a[idx] - b[idx]
            idx += grid_stride
    elif dtype == DType.float32:
        # width = 4
        var base_idx = tid * 4
        var idx = base_idx
        while idx + 4 <= size:
            var a_vec = a.load[width=4](idx)
            var b_vec = b.load[width=4](idx)
            var result = a_vec - b_vec
            output.store[width=4](idx, result)
            idx += grid_stride * 4
        while idx < size:
            output[idx] = a[idx] - b[idx]
            idx += grid_stride
    elif dtype == DType.float64:
        # width = 2
        var base_idx = tid * 2
        var idx = base_idx
        while idx + 2 <= size:
            var a_vec = a.load[width=2](idx)
            var b_vec = b.load[width=2](idx)
            var result = a_vec - b_vec
            output.store[width=2](idx, result)
            idx += grid_stride * 2
        while idx < size:
            output[idx] = a[idx] - b[idx]
            idx += grid_stride
    else:
        # Scalar fallback with grid-stride
        var idx = tid
        while idx < size:
            output[idx] = a[idx] - b[idx]
            idx += grid_stride


fn matrix_mul_kernel_vectorized[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: UInt,
    total_threads: UInt,
):
    """Optimized GPU kernel for element-wise matrix multiplication with vectorization.

    Features:
    - Vectorized memory operations (dtype-specific widths)
    - Coalesced memory access
    - Grid-stride loops

    Args:
        output: Output buffer for result.
        a: First input matrix buffer.
        b: Second input matrix buffer.
        size: Total number of elements.
        total_threads: Total number of threads launched (grid_size * block_size).
    """
    var tid = block_dim.x * block_idx.x + thread_idx.x
    var grid_stride = total_threads

    @parameter
    if dtype == DType.float16:
        var base_idx = tid * 8
        var idx = base_idx
        while idx + 8 <= size:
            var a_vec = a.load[width=8](idx)
            var b_vec = b.load[width=8](idx)
            var result = a_vec * b_vec
            output.store[width=8](idx, result)
            idx += grid_stride * 8
        while idx < size:
            output[idx] = a[idx] * b[idx]
            idx += grid_stride
    elif dtype == DType.float32:
        var base_idx = tid * 4
        var idx = base_idx
        while idx + 4 <= size:
            var a_vec = a.load[width=4](idx)
            var b_vec = b.load[width=4](idx)
            var result = a_vec * b_vec
            output.store[width=4](idx, result)
            idx += grid_stride * 4
        while idx < size:
            output[idx] = a[idx] * b[idx]
            idx += grid_stride
    elif dtype == DType.float64:
        var base_idx = tid * 2
        var idx = base_idx
        while idx + 2 <= size:
            var a_vec = a.load[width=2](idx)
            var b_vec = b.load[width=2](idx)
            var result = a_vec * b_vec
            output.store[width=2](idx, result)
            idx += grid_stride * 2
        while idx < size:
            output[idx] = a[idx] * b[idx]
            idx += grid_stride
    else:
        var idx = tid
        while idx < size:
            output[idx] = a[idx] * b[idx]
            idx += grid_stride


fn matrix_fill_kernel_vectorized[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype]],
    value: Scalar[dtype],
    size: UInt,
    total_threads: UInt,
):
    """Optimized GPU kernel for filling matrix with a value.

    Features:
    - Vectorized stores for better memory bandwidth (dtype-specific widths)
    - Grid-stride loops
    - Coalesced memory writes

    Args:
        output: Output buffer to fill.
        value: Value to fill with.
        size: Total number of elements.
        total_threads: Total number of threads launched (grid_size * block_size).
    """
    var tid = block_dim.x * block_idx.x + thread_idx.x
    var grid_stride = total_threads

    @parameter
    if dtype == DType.float16:
        # width = 8
        alias VecType = SIMD[dtype, 8]
        var vec_value = VecType(value)
        var base_idx = tid * 8
        var idx = base_idx
        while idx + 8 <= size:
            output.store[width=8](idx, vec_value)
            idx += grid_stride * 8
        while idx < size:
            output[idx] = value
            idx += grid_stride
    elif dtype == DType.float32:
        # width = 4
        alias VecType = SIMD[dtype, 4]
        var vec_value = VecType(value)
        var base_idx = tid * 4
        var idx = base_idx
        while idx + 4 <= size:
            output.store[width=4](idx, vec_value)
            idx += grid_stride * 4
        while idx < size:
            output[idx] = value
            idx += grid_stride
    elif dtype == DType.float64:
        # width = 2
        alias VecType = SIMD[dtype, 2]
        var vec_value = VecType(value)
        var base_idx = tid * 2
        var idx = base_idx
        while idx + 2 <= size:
            output.store[width=2](idx, vec_value)
            idx += grid_stride * 2
        while idx < size:
            output[idx] = value
            idx += grid_stride
    else:
        var idx = tid
        while idx < size:
            output[idx] = value
            idx += grid_stride


fn matrix_reduce_sum_kernel[
    dtype: DType
](
    input: UnsafePointer[Scalar[dtype]],
    output: UnsafePointer[Scalar[dtype]],
    size: UInt,
):
    """Optimized reduction sum kernel using warp primitives.

    This kernel computes a per-block reduction (one output element per
    block) by performing:
      1) a grid-stride accumulation per thread,
      2) a warp-level reduction (fast shuffle) to produce one partial sum
         per warp,
      3) writing warp partials to shared memory,
      4) a single-thread final accumulation of warp partials and write-out.

    Args:
        input: Input buffer to reduce.
        output: Output buffer for per-block sums (indexed by blockIdx.x).
        size: Total number of elements.
    """
    var tid = block_dim.x * block_idx.x + thread_idx.x
    var grid_stride = block_dim.x * grid_dim.x

    var thread_sum = Scalar[dtype](0)
    var idx = tid
    while idx < size:
        thread_sum += input[idx]
        idx += grid_stride

    var lane_id: UInt = thread_idx.x % UInt(WARP_SIZE)
    var warp_result = warp_sum(thread_sum)

    var num_warps_per_block: UInt = (block_dim.x + UInt(WARP_SIZE) - 1) // UInt(
        WARP_SIZE
    )

    var warp_shared = stack_allocation[
        MAX_WARPS_PER_BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    if lane_id == 0:
        var warp_id_in_block: UInt = thread_idx.x // UInt(WARP_SIZE)
        warp_shared[warp_id_in_block] = warp_result

    barrier()
    if thread_idx.x == 0:
        var block_total = Scalar[dtype](0)
        var i: UInt = 0
        while i < num_warps_per_block:
            block_total += warp_shared[i]
            i += 1
        output[block_idx.x] = block_total

# TODO: move to compile time and specialize based on dtype.
fn calculate_1d_launch_params_for_dtype[
    dtype: DType
](total_elements: Int) -> Tuple[Int, Int]:
    """Calculate 1D launch parameters specialized for dtype, with warp alignment
    and vector-width-aware grid sizing to reduce kernel launch overhead on small inputs.
    """

    # For Apple Metal
    var block_size = 32
    var grid_size = (total_elements + block_size - 1) // block_size

    if grid_size > 8192:
        grid_size = 8192

    return (grid_size, block_size)
