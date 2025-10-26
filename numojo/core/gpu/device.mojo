from gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from gpu import thread_idx, block_dim, block_idx, grid_dim, barrier
from gpu.host.dim import Dim
from gpu.memory import AddressSpace
from memory import stack_allocation
from collections.optional import Optional
from sys import simd_width_of
from testing.testing import assert_true
from sys.info import (
    has_accelerator,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
)

alias cpu = Device.CPU
alias gpu = Device.GPU
alias cuda = Device.CUDA
alias rocm = Device.ROCM
alias metal = Device.MPS
alias mps = Device.MPS


# Device descriptor
struct Device(ImplicitlyCopyable, Movable, Representable, Stringable, Writable):
    """Execution device for arrays/matrices.

    Fields:
    - type: "cpu" | "gpu"
    - name: backend identifier ("" for CPU, "cuda" | "rocm" | "metal" for GPU)
    - id:   device index on that backend (0-based)

    Aliases:
    - CPU       -> CPU execution
    - GPU       -> Generic GPU "auto" selector (prefers available accelerator)
    - CUDA      -> NVIDIA CUDA GPU
    - ROCM      -> AMD ROCm GPU
    - METAL     -> Apple Metal GPU
    """

    var type: String
    var name: String
    var id: Int

    # Aliases for convenience
    alias CPU = Device(type="cpu", name="", id=0)
    alias GPU = Device(type="gpu", name="auto", id=0)

    alias CUDA = Device(type="gpu", name="cuda", id=0)
    alias ROCM = Device(type="gpu", name="rocm", id=0)
    alias MPS = Device(type="gpu", name="mps", id=0)

    @parameter
    @staticmethod
    fn detect_accelerator() -> String:
        """Detect the best available accelerator on the system."""

        @parameter
        if has_nvidia_gpu_accelerator():
            return "cuda"
        if has_amd_gpu_accelerator():
            return "rocm"
        if has_apple_gpu_accelerator():
            return "mps"
        return "cpu"

    fn __init__(out self, type: String, name: String, id: Int):
        try:
            assert_true(
                type == "cpu" or type == "gpu",
                "Device type must be 'cpu' or 'gpu'",
            )
            if type == "cpu":
                assert_true(name == "", "CPU device name must be empty string")
                assert_true(id == 0, "CPU device id must be 0")
            else:
                assert_true(
                    name == "cuda"
                    or name == "rocm"
                    or name == "mps"
                    or name == "auto",
                    "Invalid GPU device name",
                )
                assert_true(id >= 0, "GPU device id must be non-negative")
            self.type = type
            self.name = name
            if self.name == "auto":
                self.name = Device.detect_accelerator()
            self.id = id
        except e:
            print("Invalid device type provided. Defaulting to CPU.")
            self.type = "cpu"
            self.name = ""
            self.id = 0

    fn __repr__(self) -> String:
        return self.__str__()

    @staticmethod
    fn default() -> Device:
        """Choose a sensible default device: prefer any available GPU, else CPU.
        """
        if has_nvidia_gpu_accelerator():
            return Device.CUDA
        if has_amd_gpu_accelerator():
            return Device.ROCM
        if has_apple_gpu_accelerator():
            return Device.MPS
        return Device.CPU

    @staticmethod
    fn gpu_default() raises -> Device:
        """Choose the best available GPU, or raise if none is available."""
        if has_nvidia_gpu_accelerator():
            return Device.CUDA
        if has_amd_gpu_accelerator():
            return Device.ROCM
        if has_apple_gpu_accelerator():
            return Device.MPS
        raise Error("No GPU accelerator available on this system")

    fn available(self) -> Bool:
        """Check if this device is available on the current system."""
        if self.type == "cpu":
            return True
        if self.name == "cuda":
            return has_nvidia_gpu_accelerator()
        if self.name == "rocm":
            return has_amd_gpu_accelerator()
        if self.name == "metal":
            return has_apple_gpu_accelerator()
        if self.type == "gpu" and self.name == "auto":
            return has_accelerator()
        return False

    @staticmethod
    @parameter
    fn available_devices() -> String:
        """List all available devices on the current system."""
        var devices_string: String = "\n"
        devices_string += (
            "  • " + String(Device.CPU) + " (Default CPU device)\n"
        )

        if has_nvidia_gpu_accelerator():
            devices_string += (
                "  • " + String(Device.CUDA) + " (NVIDIA CUDA GPU)\n"
            )
        if has_amd_gpu_accelerator():
            devices_string += "  • " + String(Device.ROCM) + " (AMD ROCm GPU)\n"
        if has_apple_gpu_accelerator():
            devices_string += (
                "  • " + String(Device.MPS) + " (Apple Metal GPU)\n"
            )

        if not (
            has_nvidia_gpu_accelerator()
            or has_amd_gpu_accelerator()
            or has_apple_gpu_accelerator()
        ):
            devices_string += "  (No GPU accelerators detected)"

        return devices_string

    fn __str__(self) -> String:
        try:
            return String("Device(type='{}', name='{}', id={})").format(
                self.type, self.name, self.id
            )
        except:
            return "Device(Invalid)"

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __eq__(self, other: Self) -> Bool:
        return (
            (self.type == other.type)
            and (self.id == other.id)
            and (self.name == other.name)
        )

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)


@parameter
fn check_accelerator[device: Device]() -> Bool:
    @parameter
    if device.type != "gpu":
        return False

    @parameter
    if device.name == "auto":
        return has_accelerator()

    @parameter
    if device.name == "" or device.name == "cpu":
        return False

    @parameter
    if device.name == "cuda":
        return has_nvidia_gpu_accelerator()
    elif device.name == "rocm":
        return has_amd_gpu_accelerator()
    elif device.name == "mps" or device.name == "metal":
        return has_apple_gpu_accelerator()
    else:
        return False
