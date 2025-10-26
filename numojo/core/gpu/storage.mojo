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

from numojo.core.gpu.device import Device, check_accelerator


struct HostStorage[dtype: DType, device: Device](
    Copyable, Movable
): # similar to owndata 
    var ptr: UnsafePointer[Scalar[dtype]]

    fn __init__(out self, size: Int):
        """
        Allocate given space on memory.
        The bytes allocated is `size` * `byte size of dtype`.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as True.
        The memory should be freed by `__del__`.
        """
        self.ptr = UnsafePointer[Scalar[dtype]]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Scalar[dtype]]):
        """
        Do not use this if you know what it means.
        If the pointer is associated with another array, it might cause
        dangling pointer problem.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as False.
        The memory should not be freed by `__del__`.
        """
        self.ptr = ptr

    fn __moveinit__(out self, deinit other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.ptr


struct DeviceStorage[dtype: DType, device: Device](Copyable, Movable):
    var buffer: DeviceBuffer[dtype]

    fn __init__(out self, size: Int) raises:
        constrained[
            has_accelerator(), "No GPU device available for DeviceStorage"
        ]()
        var buf = DeviceContext().enqueue_create_buffer[dtype](size)
        self.buffer = buf^

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^

    fn get_buffer(self) -> DeviceBuffer[dtype]:
        return self.buffer

    fn get_device_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.buffer.unsafe_ptr()

    fn get_device_context(self) raises -> DeviceContext:
        return self.buffer.context()


struct DataContainer[dtype: DType, device: Device = Device.CPU](
    Copyable, Movable
):
    """Storage container using Optional approach for CPU/GPU memory.

    Performance characteristics:
    - Memory: Only allocates space for the storage type being used
    - Access: Direct .value() call, no pattern matching needed
    - Type safety: Clear which storage type is active
    """

    var host_storage: Optional[HostStorage[dtype, device]]
    var device_storage: Optional[DeviceStorage[dtype, device]]
    var size: Int

    fn __init__(out self, size: Int) raises:
        self.size = size

        @parameter
        if device.type == "cpu":
            self.host_storage = HostStorage[dtype, device](size)
            self.device_storage = None
        elif device.type == "gpu":
            if not check_accelerator[device]():
                raise Error(
                    "\n Requested GPU device: "
                    + String(device)
                    + " is not available. The available devices are: "
                    + Device.available_devices()
                )
            self.host_storage = None
            self.device_storage = DeviceStorage[dtype, device](size)
        else:
            raise Error("Unsupported device type: " + String(device.type))

    fn __moveinit__(out self, deinit other: Self):
        self.host_storage = other.host_storage^
        self.device_storage = other.device_storage^
        self.size = other.size

    fn get_host_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        """Get pointer for CPU access. For GPU, use map_to_host context."""
        constrained[
            device == Device.CPU,
            "Cannot retrieve a pointer to Device memory for a HostBuffer",
        ]()
        return self.host_storage.unsafe_value().get_ptr()

    fn get_device_ptr(self) raises -> UnsafePointer[Scalar[dtype]]:
        """Get pointer for GPU access."""
        constrained[
            (
                device == Device.GPU
                or device == Device.CUDA
                or device == Device.ROCM
                or device == Device.MPS
            ),
            "Cannot retrieve a pointer to Host memory for a DeviceBuffer",
        ]()
        return self.device_storage.value().get_device_ptr()

    @parameter
    fn is_cpu(self) -> Bool:
        return device.type == "cpu"

    @parameter
    fn is_gpu(self) -> Bool:
        return device.type == "gpu"

    fn get_device_buffer(self) raises -> DeviceBuffer[dtype]:
        constrained[
            (
                device == Device.GPU
                or device == Device.CUDA
                or device == Device.ROCM
                or device == Device.MPS
            ),
            "Cannot retrieve a DeviceBuffer for a HostBuffer",
        ]()
        return self.device_storage.value().get_buffer()
