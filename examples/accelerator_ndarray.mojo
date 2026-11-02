# ===----------------------------------------------------------------------=== #
# NuMojo: AcceleratorNDArray owner, view, and trait examples
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
AcceleratorNDArray examples (examples.accelerator_ndarray).
===========================================================
Demonstrates CPU and GPU owners, borrowed views, and generic `NDArrayLike`
functions.

Exports
-------
- `show_array`: Prints metadata shared by owners and views.
- `first_four`: Loads four contiguous CPU elements through generic access.
- `cpu_examples`: Demonstrates CPU ownership and borrowing.
- `gpu_example`: Demonstrates one available GPU backend.
- `main`: Runs CPU examples and a guarded GPU example.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.sys.info import default_accelerator

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.array_like import NDArrayLike
from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.layout.ndlayout import Shape


def show_array[A: NDArrayLike](ref array: A) raises:
    """Print device information for any accelerator array or view.

    Parameters:
        A: Owner or view type implementing `NDArrayLike`.

    Args:
        array: Array owner or borrowed view to inspect.

    Raises:
        NumojoError: If the first CPU element cannot be read.

    Examples:
        `show_array(owner)` and `show_array(owner.view())` use one function.
    """
    print(
        "device:",
        A.DeviceType.device_name(),
        "shape:",
        array.shape().tolist(),
        "strides:",
        array.strides().tolist(),
        "size:",
        array.size(),
        "offset:",
        array.offset(),
        "owns data:",
        array.owns_data(),
        "writeable:",
        array.is_writeable(),
    )
    comptime if A.DeviceType.type == "cpu":
        print(
            "first element:",
            array.unsafe_ptr().unsafe_load[width=1](0)[0],
        )


def first_four[
    A: NDArrayLike
](ref array: A,) -> SIMD[A.ElementType, 4]:
    """Load four contiguous elements through generic pointer access.

    Parameters:
        A: CPU owner or view type implementing `NDArrayLike`.

    Args:
        array: C-contiguous array with at least four elements.

    Returns:
        First four physical elements as one SIMD value.

    Examples:
        `first_four(owner)` and `first_four(owner.view())` share this code.
    """
    comptime assert A.DeviceType.type == "cpu", "first_four is CPU-only"
    return array.unsafe_ptr().unsafe_load[width=4](0)


def _set_first_through_view(
    mut owner: AcceleratorNDArray[DType.float32, Device.CPU],
) raises:
    """Write through a mutable view whose borrow ends on return."""
    var writable = owner.view_mut()
    writable.itemset(0, 42.0)


def cpu_examples() raises:
    """Demonstrate CPU owners, immutable views, mutable views, and copies.

    Raises:
        NumojoError: If an array or layout operation fails.

    Examples:
        Call `cpu_examples()` from `main()`.
    """
    print("--- CPU owner and immutable views ---")
    var owner = AcceleratorNDArray[DType.float32, Device.CPU](Shape(2, 4))
    for i in range(owner.size()):
        owner.itemset(i, Float32(i + 1))

    show_array(owner)
    print("first four owner values:", first_four(owner))

    var transposed = owner.view().transpose()
    show_array(transposed)
    print("transposed[3, 1]:", transposed.item(3, 1))

    var strided = owner.view().reshape(Shape(8)).slice(0, 0, 4, 2)
    show_array(strided)
    print(
        "strided values:",
        strided.item(0),
        strided.item(1),
        strided.item(2),
        strided.item(3),
    )

    print("--- Mutable view ---")
    _set_first_through_view(owner)
    print("owner after view write:", owner.item(0))

    print("--- Independent owner copy ---")
    var copied = owner.copy()
    copied.itemset(0, -1.0)
    print("owner[0]:", owner.item(0), "copy[0]:", copied.item(0))


def gpu_example[device: Device]() raises:
    """Transfer arrays to one GPU backend and run generic and concrete APIs.

    Parameters:
        device: Available CUDA, ROCm, or MPS device.

    Raises:
        NumojoError: If allocation, transfer, or execution fails.

    Examples:
        Call `gpu_example[Device.MPS]()` on an Apple GPU.
    """
    comptime assert device.type == "gpu", "gpu_example requires a GPU device"
    print("--- GPU:", device.device_name(), "---")
    var host = zeros[DType.float32, Device.CPU](Shape(8))
    for i in range(host.size()):
        host.itemset(i, Float32(i + 1))

    var left = host.to[device]()
    var right = host.to[device]()
    show_array(left)
    show_array(left.view())

    var result = left + right
    var result_host = result.to_host()
    print("GPU result on host:", result_host)
    print("GPU sum:", result.sum())


def main() raises:
    """Run all examples, selecting the available GPU backend if present.

    Raises:
        NumojoError: If an example operation fails.

    Examples:
        Run with `pixi run mojo run -I . examples/accelerator_ndarray.mojo`.
    """
    cpu_examples()

    comptime if default_accelerator().is_nvidia_gpu():
        gpu_example[Device.CUDA]()
    elif default_accelerator().is_amd_gpu():
        gpu_example[Device.ROCM]()
    elif default_accelerator().is_apple_gpu():
        gpu_example[Device.MPS]()
    else:
        print("No GPU accelerator available; GPU example skipped.")
