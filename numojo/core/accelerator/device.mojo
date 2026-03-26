# ===----------------------------------------------------------------------=== #
# NuMojo: Device struct for CPU/GPU execution
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Device (numojo.core.accelerator.device)

This module defines the `Device` struct, which represents an execution device for array and matrix operations.
It supports CPU and GPU devices, with GPU backends for NVIDIA CUDA, AMD ROCm, and Apple Metal.
"""
from std.sys.info import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
)

from numojo.core.error import NumojoError

comptime cpu = Device.CPU
comptime cuda = Device.CUDA
comptime rocm = Device.ROCM
comptime mps = Device.MPS


struct Device(
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Writable,
):
    """Represents an execution device for array and matrix operations.

    A `Device` identifies where computation should run, analogous to
    `torch.device` in PyTorch. Each device has a type ("cpu" or "gpu"),
    an optional backend name ("cuda", "rocm", or "mps" for GPUs), and
    a zero-based device index.

    Use the predefined comptime constants for common devices:
        - `Device.CPU`  — CPU execution.
        - `Device.CUDA` — NVIDIA CUDA GPU.
        - `Device.ROCM` — AMD ROCm GPU.
        - `Device.MPS`  — Apple Metal GPU.

    Devices can also be constructed from torch-style strings:
        ```
        var dev = Device("cuda:0")
        var cpu = Device("cpu")
        ```
    """

    var type: String
    """Device type: "cpu" or "gpu"."""
    var name: String
    """Backend identifier: "" for CPU, "cuda" | "rocm" | "mps" for GPU."""
    var id: Int
    """Zero-based device index on the backend."""

    # ===------------------------------------------------------------------=== #
    # Comptime device constants
    # ===------------------------------------------------------------------=== #

    comptime CPU = Device._unchecked_init(type="cpu", name="", id=0)
    """CPU device."""
    comptime CUDA = Device._unchecked_init(type="gpu", name="cuda", id=0)
    """NVIDIA CUDA GPU device."""
    comptime ROCM = Device._unchecked_init(type="gpu", name="rocm", id=0)
    """AMD ROCm GPU device."""
    comptime MPS = Device._unchecked_init(type="gpu", name="mps", id=0)
    """Apple Metal GPU device."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    def __init__(out self):
        """Initialize a default CPU device."""
        self.type = "cpu"
        self.name = ""
        self.id = 0

    def __init__(out self, text: String) raises:
        """Initialize a device by parsing a torch-style device string.

        Supported formats: "cpu", "cuda", "cuda:0", "rocm", "rocm:1",
        "mps", "mps:0", "gpu".

        Args:
            text: A device string to parse.

        Raises:
            Error on invalid device string format.
        """
        var parsed = Device.parse_device_string(text)
        self.type = parsed.type
        self.name = parsed.name
        self.id = parsed.id

    def __init__(out self, type: String, name: String, id: Int):
        """Initialize a device with explicit type, name, and index.

        Validates the arguments and falls back to CPU if the requested
        GPU backend is not available on the current system.

        Args:
            type: Device type, must be "cpu" or "gpu".
            name: Backend name ("" for CPU; "cuda", "rocm", or "mps" for GPU).
            id: Zero-based device index (must be 0 for CPU, >= 0 for GPU).
        """
        if type != "cpu" and type != "gpu":
            print("Invalid device type '" + type + "'. Defaulting to CPU.")
            self = Device._cpu_fallback()
            return

        if type == "gpu" and name == "":
            self = Device._cpu_fallback()
            return

        if type == "cpu":
            if name != "":
                print("CPU device name must be empty. Defaulting to CPU.")
                self = Device._cpu_fallback()
                return
            if id != 0:
                print("CPU device id must be 0. Defaulting to CPU.")
                self = Device._cpu_fallback()
                return
            self.type = "cpu"
            self.name = ""
            self.id = 0
            return

        if name != "cuda" and name != "rocm" and name != "mps":
            print("Invalid GPU backend '" + name + "'. Defaulting to CPU.")
            self = Device._cpu_fallback()
            return

        if id < 0:
            print("GPU device id must be non-negative. Defaulting to CPU.")
            self = Device._cpu_fallback()
            return

        if name == "cuda" and not has_nvidia_gpu_accelerator():
            self = Device._cpu_fallback()
            return
        if name == "rocm" and not has_amd_gpu_accelerator():
            self = Device._cpu_fallback()
            return
        if name == "mps" and not has_apple_gpu_accelerator():
            self = Device._cpu_fallback()
            return

        self.type = type
        self.name = name
        self.id = id

    @staticmethod
    def _unchecked_init(
        out device: Device, type: String, name: String, id: Int
    ):
        """Create a device without any validation. For internal/comptime use."""
        device = Device()
        device.type = type
        device.name = name
        device.id = id

    @staticmethod
    def _cpu_fallback() -> Device:
        """Return a default CPU device."""
        return Device()

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    def __str__(self) -> String:
        """Return a human-readable string representation.

        Returns:
            A string like `Device(type='cpu', name='', id=0)`.
        """
        return (
            "Device(type='"
            + self.type
            + "', name='"
            + self.name
            + "', id="
            + String(self.id)
            + ")"
        )

    def __repr__(self) -> String:
        """Return the canonical string representation.

        Returns:
            Same as `__str__`.
        """
        # TODO: repr is deprecated in favor of write_repr_to
        return self.__str__()

    def write_repr_to[W: Writer](self, mut writer: W):
        """Write the string representation to a writer.

        Parameters:
            W: The writer type.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.__str__())

    def write_to[W: Writer](self, mut writer: W):
        """Write the string representation to a writer.

        Parameters:
            W: The writer type.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.__str__())

    def __eq__(self, other: Self) -> Bool:
        """Check equality with another device.

        Args:
            other: The device to compare against.

        Returns:
            True if type, name, and id all match.
        """
        return (
            self.type == other.type
            and self.name == other.name
            and self.id == other.id
        )

    def __ne__(self, other: Self) -> Bool:
        """Check inequality with another device.

        Args:
            other: The device to compare against.

        Returns:
            True if the devices differ in any field.
        """
        return not self.__eq__(other)

    # ===------------------------------------------------------------------=== #
    # Instance methods
    # ===------------------------------------------------------------------=== #

    def is_cpu(self) -> Bool:
        """Check if this is a CPU device.

        Returns:
            True if the device type is "cpu".
        """
        return self.type == "cpu"

    def is_gpu(self) -> Bool:
        """Check if this is a GPU device.

        Returns:
            True if the device type is "gpu".
        """
        return self.type == "gpu"

    def is_available(self) -> Bool:
        """Check if this device is available on the current system.

        Returns:
            True if the device hardware is present. CPU always returns True.
        """
        if self.type == "cpu":
            return True
        if self.type == "gpu":
            if self.name == "cuda":
                return has_nvidia_gpu_accelerator()
            elif self.name == "rocm":
                return has_amd_gpu_accelerator()
            elif self.name == "mps":
                return has_apple_gpu_accelerator()
        return False

    # ===------------------------------------------------------------------=== #
    # Static methods
    # ===------------------------------------------------------------------=== #

    @staticmethod
    def default_device() -> Device:
        """Return the best available device: GPU if present, otherwise CPU.

        Returns:
            A `Device` for the first available GPU backend, or `Device.CPU`.
        """
        var backend: String
        try:
            backend = Device.available_gpu()
        except:
            return Device.CPU

        return Device(type="gpu", name=backend, id=0)

    @staticmethod
    @parameter
    def available_gpu() raises -> String:
        """Return the name of the best available GPU backend.

        Checks in order: CUDA → ROCm → MPS.

        Returns:
            "cuda", "rocm", or "mps" depending on available hardware.

        Raises:
            NumojoError if no GPU accelerator is detected.
        """

        comptime if has_nvidia_gpu_accelerator():
            return "cuda"

        comptime if has_amd_gpu_accelerator():
            return "rocm"

        comptime if has_apple_gpu_accelerator():
            return "mps"
        else:
            raise NumojoError(
                category="value",
                message="No GPU accelerators detected.",
                location="Device.available_gpu()",
            )

    @staticmethod
    @parameter
    def available_devices() -> String:
        """List all available devices on the current system.

        Returns:
            A formatted multi-line string of available devices.
        """
        var result: String = "\n"
        result += "  • " + String(Device.CPU) + " (Default CPU device)\n"

        comptime if has_nvidia_gpu_accelerator():
            result += "  • " + String(Device.CUDA) + " (NVIDIA CUDA GPU)\n"

        comptime if has_amd_gpu_accelerator():
            result += "  • " + String(Device.ROCM) + " (AMD ROCm GPU)\n"

        comptime if has_apple_gpu_accelerator():
            result += "  • " + String(Device.MPS) + " (Apple Metal GPU)\n"

        comptime if not (
            has_nvidia_gpu_accelerator()
            or has_amd_gpu_accelerator()
            or has_apple_gpu_accelerator()
        ):
            result += "  (No GPU accelerators detected)"

        return result

    @staticmethod
    @parameter
    def parse_device_string(text: String) raises -> Device:
        """Parse a torch-style device string into a `Device`.

        Supported formats:
            - "cpu"
            - "cuda", "cuda:0", "cuda:1", ...
            - "rocm", "rocm:0", "rocm:1", ...
            - "mps", "mps:0", "mps:1", ...
            - "gpu" (resolves to best available GPU backend)

        Args:
            text: The device string to parse.

        Returns:
            The parsed `Device`. Falls back to `Device.CPU` for
            unrecognized or invalid strings.

        Raises:
            Error when "gpu" is specified but no GPU backend is available.
        """
        if text == "cpu" or text.startswith("cpu:"):
            return Device.CPU

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
            backend = Device.available_gpu()

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


# ===----------------------------------------------------------------------=== #
# Module-level functions
# ===----------------------------------------------------------------------=== #


@parameter
def is_accelerator_available[device: Device]() -> Bool:
    """Check at compile time whether the given device's GPU accelerator exists.

    Parameters:
        device: The device to check.

    Returns:
        True if the device is a GPU and its backend hardware is available.
        Always returns False for CPU or unknown backends.
    """

    comptime if device.type != "gpu":
        return False

    comptime if device.name == "":
        return False

    comptime if device.name == "cuda":
        return has_nvidia_gpu_accelerator()
    elif device.name == "rocm":
        return has_amd_gpu_accelerator()
    elif device.name == "mps":
        return has_apple_gpu_accelerator()
    else:
        return False
