from sys.info import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
)
from testing import assert_true

comptime cpu = Device.CPU
"""CPU device alias for convenience."""
comptime gpu = Device.GPU
"""Generic GPU device alias for convenience (selects best available GPU)."""
comptime cuda = Device.CUDA
"""NVIDIA CUDA GPU device alias for convenience."""
comptime rocm = Device.ROCM
"""AMD ROCm GPU device alias for convenience."""
comptime mps = Device.MPS
"""Apple Metal GPU device alias for convenience."""

struct Device(ImplicitlyCopyable, Movable, Representable, Stringable, Writable):
    """Execution device for arrays/matrices.

    Fields:
    - type: "cpu" | "gpu"
    - name: backend identifier ("" for CPU, "cuda" | "rocm" | "mps" for GPU)
    - id:   device index on that backend (0-based)

    Comptimes:
    - CPU       -> CPU execution
    - GPU       -> Best available GPU (falls back to CPU if none)
    - CUDA      -> NVIDIA CUDA GPU
    - ROCM      -> AMD ROCm GPU
    - MPS       -> Apple Metal GPU
    """

    var type: String
    var name: String
    var id: Int

    comptime CPU = Device(type="cpu", name="", id=0)
    comptime GPU = Device(type="gpu", name=Device.preferred_gpu_backend(), id=0)

    comptime CUDA = Device(type="gpu", name="cuda", id=0)
    comptime ROCM = Device(type="gpu", name="rocm", id=0)
    comptime MPS = Device(type="gpu", name="mps", id=0)

    @parameter
    @staticmethod
    fn preferred_gpu_backend() -> String:
        """Return best available GPU backend name, or "" if none."""
        @parameter
        if has_nvidia_gpu_accelerator():
            return "cuda"
        if has_amd_gpu_accelerator():
            return "rocm"
        if has_apple_gpu_accelerator():
            return "mps"
        return ""

    @parameter
    @staticmethod
    fn parse_device_string(text: String) -> Device:
        """Parse torch-like device strings.

        Supported:
        - "cpu"
        - "cuda", "cuda:0"
        - "rocm", "rocm:0"
        - "mps", "mps:0"
        - "gpu" (best available GPU, or CPU if none)
        """
        if text == "cpu":
            return Device.CPU
        if text == "gpu":
            var backend = Device.preferred_gpu_backend()
            if backend == "":
                return Device.CPU
            return Device(type="gpu", name=backend, id=0)

        var backend: String = ""
        var id: Int = 0
        var seen_colon: Bool = False
        var id_str: String = ""
        for ch in text.codepoint_slices():
            if not seen_colon and ch == ":":
                seen_colon = True
                continue
            if not seen_colon:
                backend += ch
            else:
                id_str += ch

        if backend == "":
            return Device.CPU

        if backend == "gpu":
            backend = Device.preferred_gpu_backend()
            if backend == "":
                return Device.CPU

        if seen_colon and id_str == "":
            return Device.CPU

        if id_str != "":
            var sign: Int = 1
            var has_digit: Bool = False
            var bytes = id_str.as_bytes()
            for i in range(len(bytes)):
                var b = bytes[i]
                if i == 0 and Int(b) == ord("-"):
                    sign = -1
                    continue
                if Int(b) < ord("0") or Int(b) > ord("9"):
                    return Device.CPU
                has_digit = True
                id = id * 10 + (Int(b) - ord("0"))
            if not has_digit:
                return Device.CPU
            id = id * sign
            if id < 0:
                return Device.CPU

        if backend == "cuda" or backend == "rocm" or backend == "mps":
            return Device(type="gpu", name=backend, id=id)

        return Device.CPU

    fn __init__(out self, text: String):
        try:
            var parsed = Device.parse_device_string(text)
            self.type = parsed.type
            self.name = parsed.name
            self.id = parsed.id
        except e:
            print("Invalid device type provided. Defaulting to CPU.")
            self.type = "cpu"
            self.name = ""
            self.id = 0

    fn __init__(out self, type: String, name: String, id: Int):
        try:
            if type == "gpu" and name == "":
                self.type = "cpu"
                self.name = ""
                self.id = 0
                return
            assert_true(
                type == "cpu" or type == "gpu",
                "Device type must be 'cpu' or 'gpu'",
            )
            if type == "cpu":
                assert_true(name == "", "CPU device name must be empty string")
                assert_true(id == 0, "CPU device id must be 0")
            else:
                assert_true(
                    name == "cuda" or name == "rocm" or name == "mps",
                    "Invalid GPU device name",
                )
                assert_true(id >= 0, "GPU device id must be non-negative")
                if name == "cuda" and not has_nvidia_gpu_accelerator():
                    self.type = "cpu"
                    self.name = ""
                    self.id = 0
                    return
                if name == "rocm" and not has_amd_gpu_accelerator():
                    self.type = "cpu"
                    self.name = ""
                    self.id = 0
                    return
                if name == "mps" and not has_apple_gpu_accelerator():
                    self.type = "cpu"
                    self.name = ""
                    self.id = 0
                    return
            self.type = type
            self.name = name
            self.id = id
        except e:
            print("Invalid device type provided. Defaulting to CPU.")
            self.type = "cpu"
            self.name = ""
            self.id = 0

    fn __repr__(self) -> String:
        return self.__str__()

    @staticmethod
    fn default_device() -> Device:
        """Choose a sensible default device: prefer any available GPU, else CPU."""
        var backend = Device.preferred_gpu_backend()
        if backend == "":
            return Device.CPU
        return Device(type="gpu", name=backend, id=0)

    @staticmethod
    fn require_gpu() raises -> Device:
        """Choose the best available GPU, or raise if none is available."""
        var backend = Device.preferred_gpu_backend()
        if backend == "":
            raise Error("No GPU accelerator available on this system")
        return Device(type="gpu", name=backend, id=0)

    fn is_available(self) -> Bool:
        """Check if this device is available on the current system."""
        if self.type == "cpu":
            return True
        if self.name == "cuda":
            return has_nvidia_gpu_accelerator()
        if self.name == "rocm":
            return has_amd_gpu_accelerator()
        if self.name == "mps":
            return has_apple_gpu_accelerator()
        return False

    @staticmethod
    @parameter
    fn list_available_devices() -> String:
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
fn is_accelerator_available[device: Device]() -> Bool:
    @parameter
    if device.type != "gpu":
        return False

    @parameter
    if device.name == "":
        return False

    @parameter
    if device.name == "cuda":
        return has_nvidia_gpu_accelerator()
    elif device.name == "rocm":
        return has_amd_gpu_accelerator()
    elif device.name == "mps":
        return has_apple_gpu_accelerator()
    else:
        return False
