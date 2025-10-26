# ===----------------------------------------------------------------------=== #
# StaticMatrix2D for CPU and GPU
# ===----------------------------------------------------------------------=== #
from time import perf_counter_ns
from gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from gpu import thread_idx, block_dim, block_idx, grid_dim, barrier
from gpu.host.dim import Dim
from gpu.memory import AddressSpace
from memory import stack_allocation
from gpu.globals import WARP_SIZE, MAX_THREADS_PER_BLOCK_METADATA
from collections.optional import Optional
from memory import UnsafePointer
from sys import simd_width_of
from sys.info import (
    has_accelerator,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
)

from numojo.core.flags import Flags
from numojo.core.own_data import OwnData
from numojo.core.error import ValueError
from numojo.core.gpu.matrix_kernels import (
    matrix_add_kernel_vectorized,
    matrix_sub_kernel_vectorized,
    matrix_mul_kernel_vectorized,
    calculate_1d_launch_params_for_dtype,
)
from testing.testing import assert_true

from numojo.core.gpu.device import Device, check_accelerator
from numojo.core.gpu.storage import DataContainer, HostStorage, DeviceStorage


struct StaticMatrix[dtype: DType = DType.float32, device: Device = Device.CPU](
    Movable, Stringable, Writable
):
    """Matrix implementation using Optional storage approach with corrected kernels.
    """

    alias width: Int = simd_width_of[dtype]()
    var _buf: DataContainer[dtype, device]
    var shape: Tuple[Int, Int]
    var size: Int
    var strides: Tuple[Int, Int]
    var flags: Flags

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        order: String = "C",
        fill_value: Scalar[dtype] = 0.0,
    ) raises:
        self.shape = (shape[0], shape[1])
        if order == "C":
            self.strides = (shape[1], 1)
        else:
            self.strides = (1, shape[0])
        self.size = shape[0] * shape[1]
        self._buf = DataContainer[dtype, device](self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )
        self.fill(fill_value)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = other._buf.copy()
        self.flags = other.flags

    fn __moveinit__(out self, deinit other: Self):
        self._buf = other._buf^
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self.flags = other.flags

    @always_inline("nodebug")
    fn __del__(deinit self):
        @parameter
        if device.type == "cpu":
            owndata = self.flags.OWNDATA and self._buf.host_storage
            if owndata:
                self._buf.host_storage.value().ptr.free()

    fn _check_bounds(self, row: Int, col: Int) raises:
        if row < 0 or row >= self.shape[0] or col < 0 or col >= self.shape[1]:
            raise Error(
                ValueError(
                    message=String(
                        "Index out of bounds: ({}, {}) for shape ({}, {})."
                    ).format(row, col, self.shape[0], self.shape[1]),
                    suggestion="Ensure 0 <= row < rows and 0 <= col < cols",
                    location="StaticMatrix._check_bounds",
                )
            )

    fn _index(self, row: Int, col: Int) -> Int:
        return row * self.strides[0] + col * self.strides[1]

    # ! perhaps we should remove indexing into gpu arrays since they are extremely slow and results in too many copies!
    fn __getitem__(self, row: Int, col: Int) raises -> Scalar[dtype]:
        self._check_bounds(row, col)
        var idx = self._index(row, col)

        @parameter
        if device.type == "cpu":
            return self._buf.get_host_ptr()[idx]
        elif device.type == "gpu":
            with self._buf.get_device_buffer().map_to_host() as host_ptr:
                return host_ptr[idx]
        else:
            raise Error("Unsupported device type for __getitem__")

    fn __getitem__(self, var x: Int) raises -> Self:
        """1D indexing to get a row as a new StaticMatrix."""
        if x < 0 or x >= self.shape[0]:
            raise Error(
                ValueError(
                    message=String(
                        "Row index out of bounds: {} for shape ({}, {})."
                    ).format(x, self.shape[0], self.shape[1]),
                    suggestion="Ensure 0 <= row < rows",
                    location="StaticMatrix.__getitem__",
                )
            )
        var row_StaticMatrix = StaticMatrix[dtype, device](
            (1, self.shape[1]), order=String("C")
        )
        for j in range(self.shape[1]):
            row_StaticMatrix[0, j] = self[x, j]
        return row_StaticMatrix^

    fn __setitem__(mut self, row: Int, col: Int, value: Scalar[dtype]) raises:
        self._check_bounds(row, col)
        var idx = self._index(row, col)

        @parameter
        if device.type == "cpu":
            self._buf.get_host_ptr()[idx] = value
        elif device.type == "gpu":
            with self._buf.get_device_buffer().map_to_host() as host_ptr:
                host_ptr[idx] = value
        else:
            raise Error("Unsupported device type for __setitem__")

    fn fill(mut self, value: Scalar[dtype]) raises:
        """Fill matrix with value - optimized per device type."""

        @parameter
        if device.type == "cpu":
            var ptr = self._buf.get_host_ptr()
            for i in range(self.size):
                ptr[i] = value
        elif (
            device.type == "gpu"
        ):  # I should create a kernel for this instead to do it safely.
            with self._buf.get_device_buffer().map_to_host() as host_ptr:
                for i in range(self.size):
                    host_ptr[i] = value
        else:
            raise Error("Unsupported device type for fill")

    fn __str__(self) -> String:
        try:
            var result = ""
            if self.size == 0:
                return result + "[]"

            if self.shape[0] > 6 or self.shape[1] > 6:
                return self._format_large_StaticMatrix()

            for i in range(self.shape[0]):
                result += "["
                for j in range(self.shape[1]):
                    var v = self[i, j]
                    result += String(v)
                    if j < self.shape[1] - 1:
                        result += ", "
                result += "]"
                if i < self.shape[0] - 1:
                    result += "\n"

            result += String("\nStaticMatrix(")
            result += (
                String(self.shape[0])
                + "x"
                + String(self.shape[1])
                + ", dtype="
                + String(dtype)
                + ", device="
                + String(device)
                + ")\n"
            )
            return result
        except:
            return "Error generating string representation of StaticMatrix."

    fn _format_large_StaticMatrix(self) -> String:
        """Format large StaticMatrixs with truncated representation and shape info.
        """
        try:
            var result = String("StaticMatrix(")
            result += (
                String(self.shape[0])
                + "x"
                + String(self.shape[1])
                + ", dtype="
                + String(dtype)
                + ", device="
                + String(device)
                + ")\n"
            )

            for i in range(min(3, self.shape[0])):
                result += "["
                for j in range(min(3, self.shape[1])):
                    result += String(self[i, j])
                    if j < min(3, self.shape[1]) - 1:
                        result += ", "
                if self.shape[1] > 3:
                    result += ", ..., " + String(self[i, self.shape[1] - 1])
                result += "]"
                if i < min(3, self.shape[0]) - 1:
                    result += "\n"

            if self.shape[0] > 3:
                result += "\n...\n"
                result += "["
                for j in range(min(3, self.shape[1])):
                    result += String(self[self.shape[0] - 1, j])
                    if j < min(3, self.shape[1]) - 1:
                        result += ", "
                if self.shape[1] > 3:
                    result += ", ..., " + String(
                        self[self.shape[0] - 1, self.shape[1] - 1]
                    )
                result += "]"

            return result
        except:
            return (
                "Error generating truncated representation of large"
                " StaticMatrix."
            )

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @staticmethod
    fn zeros(rows: Int, cols: Int) raises -> StaticMatrix[dtype, device]:
        return StaticMatrix[dtype, device](
            shape=(rows, cols), fill_value=Scalar[dtype](0)
        )

    @staticmethod
    fn ones(rows: Int, cols: Int) raises -> StaticMatrix[dtype, device]:
        return StaticMatrix[dtype, device](
            shape=(rows, cols), fill_value=Scalar[dtype](1)
        )

    @staticmethod
    fn identity(size: Int) raises -> StaticMatrix[dtype, device]:
        var mat = StaticMatrix[dtype, device](
            shape=(size, size), fill_value=Scalar[dtype](0)
        )
        for i in range(size):
            mat[i, i] = Scalar[dtype](1)
        return mat^


    fn __add__(self, other: Self) raises -> Self:
        """Element-wise addition of two matrices."""
        if self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
            raise Error(
                ValueError(
                    message=String(
                        "Matrix shapes ({}, {}) and ({}, {}) are incompatible"
                        " for addition."
                    ).format(
                        self.shape[0],
                        self.shape[1],
                        other.shape[0],
                        other.shape[1],
                    ),
                    suggestion="Ensure both matrices have the same dimensions",
                    location="StaticMatrix.__add__",
                )
            )

        if self.device != other.device:
            raise Error(
                ValueError(
                    message="Cannot add matrices on different devices",
                    suggestion="Move matrices to the same device before adding",
                    location="StaticMatrix.__add__",
                )
            )

        return add[dtype, device](self, other)

    fn __sub__(self, other: Self) raises -> Self:
        """Element-wise subtraction of two matrices."""
        if self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
            raise Error(
                ValueError(
                    message=String(
                        "Matrix shapes ({}, {}) and ({}, {}) are incompatible"
                        " for subtraction."
                    ).format(
                        self.shape[0],
                        self.shape[1],
                        other.shape[0],
                        other.shape[1],
                    ),
                    suggestion="Ensure both matrices have the same dimensions",
                    location="StaticMatrix.__sub__",
                )
            )

        if self.device != other.device:
            raise Error(
                ValueError(
                    message="Cannot subtract matrices on different devices",
                    suggestion="Move matrices to the same device before subtracting",
                    location="StaticMatrix.__sub__",
                )
            )



    # Element-wise multiplication
    fn __mul__(self, other: Self) raises -> Self:
        """Element-wise multiplication of two matrices."""
        if self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
            raise Error(
                ValueError(
                    message=String(
                        "Matrix shapes ({},t {}) and ({}, {}) are incompatible"
                        " for element-wise multiplication."
                    ).format(
                        self.shape[0],
                        self.shape[1],
                        other.shape[0],
                        other.shape[1],
                    ),
                    suggestion="Ensure both matrices have the same dimensions",
                    location="StaticMatrix.__mul__",
                )
            )

        if self._buf.device != other._buf.device:
            raise Error(
                ValueError(
                    message="Cannot multiply matrices on different devices",
                    suggestion=(
                        "Move matrices to the same device before multiplying"
                    ),
                    location="StaticMatrix.__mul__",
                )
            )

        if self._buf.is_cpu():
            return self._mul_cpu(other)
        else:
            return self._mul_gpu(other)

    fn __matmul__(self, other: Self) raises -> Self:
        """Matrix multiplication of two matrices."""
        if self.shape[1] != other.shape[0]:
            raise Error(
                ValueError(
                    message=String(
                        "Matrix shapes ({}, {}) and ({}, {}) are incompatible"
                        " for matrix multiplication."
                    ).format(
                        self.shape[0],
                        self.shape[1],
                        other.shape[0],
                        other.shape[1],
                    ),
                    suggestion=(
                        "Ensure first matrix columns equal second matrix rows"
                    ),
                    location="StaticMatrix.__matmul__",
                )
            )

        if self.device != other.device:
            raise Error(
                ValueError(
                    message="Cannot multiply matrices on different devices",
                    suggestion=(
                        "Move matrices to the same device before multiplying"
                    ),
                    location="StaticMatrix.__matmul__",
                )
            )

        return matmul[dtype, device](self, other)


    fn _mul_cpu(self, other: Self) raises -> Self:
        """CPU implementation of element-wise multiplication with vectorization.
        """
        var result = StaticMatrix[dtype, device]((self.shape[0], self.shape[1]))
        var n = self.size
        var self_ptr = self._buf.get_host_ptr()
        var other_ptr = other._buf.get_host_ptr()
        var result_ptr = result._buf.get_host_ptr()

        alias simd_width = simd_width_of[dtype]()

        # TODO: use vectorize function.
        for i in range(0, n, simd_width):
            var remaining = min(simd_width, n - i)
            if remaining == simd_width:
                var a_simd = self_ptr.load[width=simd_width](i)
                var b_simd = other_ptr.load[width=simd_width](i)
                result_ptr.store[width=simd_width](i, a_simd * b_simd)
            else:
                # Handle remaining elements
                for j in range(remaining):
                    result_ptr[i + j] = self_ptr[i + j] * other_ptr[i + j]

        return result^

    fn _mul_gpu(self, other: Self) raises -> Self:
        """GPU implementation of element-wise multiplication with optimized vectorized kernel.
        """
        var result = StaticMatrix[dtype, device]((self.shape[0], self.shape[1]))
        var (grid_size, block_size) = calculate_1d_launch_params_for_dtype[
            dtype
        ](self.size)
        var grid_dim = Dim(grid_size)
        var block_dim = Dim(block_size)
        var total_threads = grid_size * block_size

        with DeviceContext() as ctx:
            var a_buf = self._buf.get_device_buffer()
            var b_buf = other._buf.get_device_buffer()
            var out_buf = result._buf.get_device_buffer()

            # Launch simple multiplication kernel
            ctx.enqueue_function[matrix_mul_kernel_vectorized[dtype]](
                out_buf.unsafe_ptr(),
                a_buf.unsafe_ptr(),
                b_buf.unsafe_ptr(),
                self.size,
                total_threads,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

            ctx.synchronize()

        return result^

    fn to[
        target_device: Device
    ](self) raises -> StaticMatrix[dtype, target_device]:
        """Return a new matrix on the target device, copying data as needed (PyTorch-like .to).
        """

        # CPU -> CPU deep copy
        @parameter
        if device.type == "cpu" and target_device.type == "cpu":
            raise Error("Redundant CPU to CPU copy attempted")
        elif device.type == "gpu" and target_device.type == "gpu":
            var out = StaticMatrix[dtype, target_device](
                (self.shape[0], self.shape[1])
            )
            with DeviceContext() as ctx:
                ctx.enqueue_copy(
                    out._buf.get_device_buffer(), self._buf.get_device_buffer()
                )
                ctx.synchronize()
            return out^
        elif device.type == "cpu" and target_device.type == "gpu":
            var out = StaticMatrix[dtype, target_device](
                (self.shape[0], self.shape[1])
            )
            with DeviceContext() as ctx:
                ctx.enqueue_copy(
                    out._buf.get_device_buffer(), self._buf.get_host_ptr()
                )
                ctx.synchronize()
            return out^
        elif device.type == "gpu" and target_device.type == "cpu":
            var out = StaticMatrix[dtype, target_device](
                (self.shape[0], self.shape[1])
            )
            with DeviceContext() as ctx:
                ctx.enqueue_copy(
                    out._buf.get_host_ptr(), self._buf.get_device_buffer()
                )
                ctx.synchronize()
            return out^

        else:
            raise Error(
                "Unsupported device type conversion: "
                + String(device)
                + " to "
                + String(target_device)
            )

    @always_inline
    fn cpu(self) raises -> StaticMatrix[dtype, Device.CPU]:
        return self.to[Device.CPU]()

    @always_inline
    fn cuda(self) raises -> StaticMatrix[dtype, Device.CUDA]:
        return self.to[Device.CUDA]()

    @always_inline
    fn rocm(self) raises -> StaticMatrix[dtype, Device.ROCM]:
        return self.to[Device.ROCM]()

    @always_inline
    fn metal(self) raises -> StaticMatrix[dtype, Device.MPS]:
        return self.to[Device.MPS]()


fn _add_cpu[
    dtype: DType,
    device: Device = Device.CPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """CPU StaticMatrix addition using vectorized operations."""
    var result = StaticMatrix[dtype, device]((a.shape[0], a.shape[1]))
    var n = a.size
    var a_ptr = a._buf.get_host_ptr()
    var b_ptr = b._buf.get_host_ptr()
    var result_ptr = result._buf.get_host_ptr()

    alias simd_width = simd_width_of[dtype]()

    for i in range(0, n, simd_width):
        var remaining = min(simd_width, n - i)
        if remaining == simd_width:
            var a_simd = a_ptr.load[width=simd_width](i)
            var b_simd = b_ptr.load[width=simd_width](i)
            result_ptr.store[width=simd_width](i, a_simd + b_simd)
        else:
            for j in range(remaining):
                result_ptr[i + j] = a_ptr[i + j] + b_ptr[i + j]

    return result^


fn _add_gpu[
    dtype: DType,
    device: Device = Device.GPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """GPU StaticMatrix addition using optimized vectorized kernel."""
    var result = StaticMatrix[dtype, device]((a.shape[0], a.shape[1]))
    var total_elems = a.size
    var (grid_size, block_size) = calculate_1d_launch_params_for_dtype[dtype](
        total_elems
    )
    var total_threads = grid_size * block_size
    var grid_dim = (grid_size, 1)
    var block_dim_1d = (block_size, 1)

    with DeviceContext() as ctx:
        ctx.enqueue_function[matrix_add_kernel_vectorized[dtype]](
            result._buf.get_device_ptr(),
            a._buf.get_device_ptr(),
            b._buf.get_device_ptr(),
            total_elems,
            total_threads,
            grid_dim=grid_dim,
            block_dim=block_dim_1d,
        )

    return result^

fn _sub_cpu[
    dtype: DType,
    device: Device = Device.CPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """CPU StaticMatrix addition using vectorized operations."""
    var result = StaticMatrix[dtype, device]((a.shape[0], a.shape[1]))
    var n = a.size
    var a_ptr = a._buf.get_host_ptr()
    var b_ptr = b._buf.get_host_ptr()
    var result_ptr = result._buf.get_host_ptr()

    alias simd_width = simd_width_of[dtype]()

    for i in range(0, n, simd_width):
        var remaining = min(simd_width, n - i)
        if remaining == simd_width:
            var a_simd = a_ptr.load[width=simd_width](i)
            var b_simd = b_ptr.load[width=simd_width](i)
            result_ptr.store[width=simd_width](i, a_simd - b_simd)
        else:
            for j in range(remaining):
                result_ptr[i + j] = a_ptr[i + j] - b_ptr[i + j]

    return result^


fn _sub_gpu[
    dtype: DType,
    device: Device = Device.GPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """GPU StaticMatrix subtraction using optimized vectorized kernel."""
    var result = StaticMatrix[dtype, device]((a.shape[0], a.shape[1]))
    var total_elems = a.size
    var (grid_size, block_size) = calculate_1d_launch_params_for_dtype[dtype](
        total_elems
    )
    var total_threads = grid_size * block_size
    var grid_dim = (grid_size, 1)
    var block_dim_1d = (block_size, 1)

    with DeviceContext() as ctx:
        ctx.enqueue_function[matrix_sub_kernel_vectorized[dtype]](
            result._buf.get_device_ptr(),
            a._buf.get_device_ptr(),
            b._buf.get_device_ptr(),
            total_elems,
            total_threads,
            grid_dim=grid_dim,
            block_dim=block_dim_1d,
        )

    return result^

fn add[
    dtype: DType, device: Device
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """StaticMatrix addition with device-specific optimized kernels."""

    @parameter
    if device.type == "cpu":
        return _add_cpu[dtype, device](a, b)
    else:
        return _add_gpu[dtype, device](a, b)

fn sub[
    dtype: DType, device: Device
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """StaticMatrix subtraction with device-specific optimized kernels."""

    @parameter
    if device.type == "cpu":
        return _sub_cpu[dtype, device](a, b)
    else:
        return _sub_gpu[dtype, device](a, b)

# ===----------------------------------------------------------------------=== #
# GPU Matrix Multiplication Kernels
# ===----------------------------------------------------------------------=== #
fn matrix_matmul_kernel_naive[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    m: Int,
    n: Int,
    k: Int,
):
    """Naive GPU matrix multiplication kernel - O(n³) with no optimizations."""
    var row: Int = block_dim.y * block_idx.y + thread_idx.y
    var col: Int = block_dim.x * block_idx.x + thread_idx.x

    if row < m and col < n:
        var acc: Scalar[dtype] = 0

        for ki in range(k):
            acc += a[row * k + ki] * b[ki * n + col]

        output[row * n + col] = acc


fn matrix_matmul_kernel_tiled[
    dtype: DType, tile_size: UInt
](
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    m: UInt,
    n: UInt,
    k: UInt,
):
    """Tiled GPU matrix multiplication kernel using shared memory tiles."""
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row: UInt = block_idx.y * tile_size + local_row
    var tiled_col: UInt = block_idx.x * tile_size + local_col

    # Allocate shared memory using stack_allocation
    var a_shared = stack_allocation[
        tile_size * tile_size,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    var acc: Scalar[dtype] = 0

    # Iterate over tiles to compute matrix product
    var num_tiles: UInt = (k + tile_size - 1) // tile_size
    for tile in range(num_tiles):
        # Load A tile - global row stays the same, col determined by tile
        if tiled_row < m and (tile * tile_size + local_col) < k:
            a_shared[local_row * tile_size + local_col] = a[
                tiled_row * k + (tile * tile_size + local_col)
            ]
        else:
            a_shared[local_row * tile_size + local_col] = 0

        # Load B tile - row determined by tile, global col stays the same
        if (tile * tile_size + local_row) < k and tiled_col < n:
            b_shared[local_row * tile_size + local_col] = b[
                (tile * tile_size + local_row) * n + tiled_col
            ]
        else:
            b_shared[local_row * tile_size + local_col] = 0

        barrier()

        # Matrix multiplication within the tile
        if tiled_row < m and tiled_col < n:
            for ki in range(tile_size):
                acc += (
                    a_shared[local_row * tile_size + ki]
                    * b_shared[ki * tile_size + local_col]
                )

        barrier()

    # Write out final result
    if tiled_row < m and tiled_col < n:
        output[tiled_row * n + tiled_col] = acc


# ===----------------------------------------------------------------------=== #
# Matrix Multiplication Functions
# ===----------------------------------------------------------------------=== #


fn _matmul_cpu[
    dtype: DType,
    device: Device = Device.CPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """CPU matrix multiplication using vectorized operations."""
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]

    var result = StaticMatrix[dtype, device]((m, n))

    var a_ptr = a._buf.get_host_ptr()
    var b_ptr = b._buf.get_host_ptr()
    var c_ptr = result._buf.get_host_ptr()

    # Initialize result
    for i in range(m * n):
        c_ptr[i] = 0

    alias simd_width = simd_width_of[dtype]()

    # Optimized loops with vectorization
    for i in range(m):
        for ki in range(k):
            var a_val = a_ptr[i * k + ki]
            var a_broadcast = SIMD[dtype, simd_width](a_val)

            var j = 0
            while j + simd_width <= n:
                var b_vec = b_ptr.load[width=simd_width](ki * n + j)
                var c_vec = c_ptr.load[width=simd_width](i * n + j)
                c_ptr.store[width=simd_width](
                    i * n + j, c_vec + a_broadcast * b_vec
                )
                j += simd_width

            while j < n:
                c_ptr[i * n + j] += a_val * b_ptr[ki * n + j]
                j += 1

    return result^


fn _matmul_gpu_naive[
    dtype: DType,
    device: Device = Device.GPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """GPU matrix multiplication using naive kernel."""
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]

    var result = StaticMatrix[dtype, device]((m, n))

    # Calculate grid and block dimensions
    alias TILE_SIZE = 16
    var blocks_x = (n + TILE_SIZE - 1) // TILE_SIZE
    var blocks_y = (m + TILE_SIZE - 1) // TILE_SIZE
    var grid_dim_2d = (blocks_x, blocks_y)
    var block_dim_2d = (TILE_SIZE, TILE_SIZE)

    # Get device buffers as UnsafePointers
    var a_ptr = a._buf.get_device_ptr()
    var b_ptr = b._buf.get_device_ptr()
    var result_ptr = result._buf.get_device_ptr()

    with DeviceContext() as ctx:
        ctx.enqueue_function[matrix_matmul_kernel_naive[dtype]](
            result_ptr,
            a_ptr,
            b_ptr,
            m,
            n,
            k,
            grid_dim=grid_dim_2d,
            block_dim=block_dim_2d,
        )

    return result^


fn _matmul_gpu_tiled[
    dtype: DType,
    device: Device = Device.GPU,
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """GPU matrix multiplication using tiled kernel."""
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]

    var result = StaticMatrix[dtype, device]((m, n))

    # Calculate grid and block dimensions
    alias TILE_SIZE = 32
    var blocks_x = (n + TILE_SIZE - 1) // TILE_SIZE
    var blocks_y = (m + TILE_SIZE - 1) // TILE_SIZE
    var grid_dim_2d = (blocks_x, blocks_y)
    var block_dim_2d = (TILE_SIZE, TILE_SIZE)

    var a_ptr = a._buf.get_device_ptr()
    var b_ptr = b._buf.get_device_ptr()
    var result_ptr = result._buf.get_device_ptr()

    with DeviceContext() as ctx:
        ctx.enqueue_function[matrix_matmul_kernel_tiled[dtype, TILE_SIZE]](
            result_ptr,
            a_ptr,
            b_ptr,
            m,
            n,
            k,
            grid_dim=grid_dim_2d,
            block_dim=block_dim_2d,
        )

    return result^


fn matmul[
    dtype: DType, device: Device
](
    a: StaticMatrix[dtype, device], b: StaticMatrix[dtype, device]
) raises -> StaticMatrix[dtype, device]:
    """Matrix multiplication with device-specific optimized kernels."""

    @parameter
    if device.type == "cpu":
        return _matmul_cpu[dtype, device](a, b)
    else:
        return _matmul_gpu_tiled[dtype, device](a, b)
