# ===----------------------------------------------------------------------=== #
# NuMojo: Storage
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Storage (numojo.core.memory.storage)

Backend storage containers for accelerator-aware data management.

This module provides three storage structs:

- `HostStorage`: Reference-counted host (CPU) memory container.
- `DeviceStorage`: Device (GPU) memory container wrapping a `DeviceBuffer`.
- `AcceleratorDataContainer`: Unified container that selects between
  `HostStorage` and `DeviceStorage` at compile time based on a `Device` parameter.
"""
from memory import UnsafePointer
from os.atomic import Atomic, Consistency, fence
from os import abort
from collections.optional import Optional
from sys.info import has_accelerator
from memory import memcpy

from gpu.host import DeviceBuffer, DeviceContext

from numojo.core.accelerator import Device
from numojo.core.accelerator.device import is_accelerator_available
from numojo.core.memory.data_container import Ownership
from numojo.core.error import NumojoError


# ===----------------------------------------------------------------------=== #
# HostStorage
# ===----------------------------------------------------------------------=== #


struct HostStorage[dtype: DType](
    Copyable & Movable & Sized & Stringable & Writable
):
    """Reference-counted host (CPU) memory container.

    Manages a contiguous buffer of `Scalar[dtype]` elements with two ownership
    modes controlled by `Ownership`:

    - **Managed**: The container owns the allocation and tracks shared
      references via an atomic reference count.  Memory is freed when the
      last reference is destroyed.
    - **External**: The container holds a non-owning view into memory
      managed elsewhere.  No reference counting or deallocation is performed.

    Parameters:
        dtype: The element type stored in the buffer.
    """

    comptime origin = MutExternalOrigin
    """Memory origin for the allocation."""

    var ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]
    """Pointer to the data array."""

    var _refcount: UnsafePointer[Atomic[DType.uint64], Self.origin]
    """Pointer to the atomic reference count (null for external containers)."""

    var ownership: Ownership
    """Ownership status of the container (Managed or External)."""

    var size: Int
    """Number of elements in the data array."""

    # ===----------------------------------------------------------------------===#
    # Constructors and Destructor
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Create an empty managed container with size 0 and refcount 1."""
        self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed
        self.size = 0

    @always_inline
    fn __init__(out self, size: Int):
        """Create a managed container with a buffer of `size` elements.

        The buffer is allocated but not initialized.  The reference count
        starts at 1.

        Args:
            size: Number of elements to allocate (must be non-negative).
        """
        if size < 0:
            abort("HostStorage: __init__() size must be non-negative")

        self.size = size
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed

        if size == 0:
            self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        else:
            self.ptr = alloc[Scalar[Self.dtype]](size)

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], Self.origin],
        size: Int,
        copy: Bool = False,
    ):
        """Create a container from an existing buffer.

        When `copy` is False the container is **external**: it stores the
        pointer as-is and will never free it.  When `copy` is True the data
        is deep-copied into a new **managed** allocation.

        Args:
            ptr: Pointer to an existing data buffer (must be non-null).
            size: Number of elements in the buffer (must be non-negative).
            copy: If True, deep-copy into owned storage; otherwise create
                  a non-owning external view.
        """
        if size < 0:
            abort("HostStorage: __init__() size must be non-negative")
        if not ptr:
            abort("HostStorage: __init__() ptr must be non-null")

        self.size = size
        if copy:
            self._refcount = alloc[Atomic[DType.uint64]](1)
            self._refcount[] = Atomic[DType.uint64](1)
            self.ptr = alloc[Scalar[Self.dtype]](size)
            memcpy(dest=self.ptr, src=ptr, count=size)
            self.ownership = Ownership.Managed
        else:
            self._refcount = UnsafePointer[Atomic[DType.uint64], Self.origin]()
            self.ptr = ptr
            self.ownership = Ownership.External

    @always_inline
    fn __init__(
        out self,
        *,
        ptr: UnsafePointer[Scalar[Self.dtype], Self.origin],
        size: Int,
        refcount: UnsafePointer[Atomic[DType.uint64], Self.origin],
        ownership: Ownership,
    ):
        """Create a HostStorage that shares an existing buffer and refcount.

        This constructor is used internally by `share()` to create a shared
        handle without allocating a new refcount. No validation is performed;
        the caller must ensure all arguments are valid.

        Args:
            ptr: Pointer to the shared data buffer.
            size: Number of elements in the buffer.
            refcount: Pointer to the shared atomic reference count.
            ownership: Ownership mode (should be Managed for shared handles).
        """
        self.ptr = ptr
        self.size = size
        self._refcount = refcount
        self.ownership = ownership

    @always_inline
    fn __copyinit__(out self, copy: Self):
        """Shallow-copy constructor.

        Copies the pointer and refcount, then atomically increments the
        reference count for managed containers.

        Args:
            copy: The source container.
        """
        self.size = copy.size
        self.ptr = copy.ptr
        self._refcount = copy._refcount
        self.ownership = copy.ownership

        if self.is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

    @always_inline
    fn __moveinit__(out self, deinit take: Self):
        """Move constructor.

        Transfers all fields without touching the reference count.

        Args:
            take: The source container (consumed).
        """
        self.ptr = take.ptr
        self._refcount = take._refcount
        self.ownership = take.ownership
        self.size = take.size

    @always_inline
    fn __del__(deinit self):
        """Destructor.

        For managed containers the reference count is atomically
        decremented.  If this was the last reference, the data buffer and
        the refcount allocation are freed.  External containers are left
        untouched.
        """
        if self.ownership == Ownership.External:
            return

        if not self.is_refcounted():
            return

        if self._refcount[].fetch_sub[ordering = Consistency.RELEASE](1) != 1:
            return

        fence[ordering = Consistency.ACQUIRE]()
        if self.ptr and self.size > 0:
            self.ptr.free()
        self._refcount.free()

    # ===----------------------------------------------------------------------===#
    # Data Access
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn unsafe_ptr(
        ref self,
    ) -> ref[self.ptr] UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Return a reference to the raw data pointer.

        Returns:
            A reference to `self.ptr`.
        """
        return self.ptr

    @always_inline
    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Return a pointer advanced by `offset` elements.

        Args:
            offset: Number of elements to advance.

        Returns:
            `self.ptr + offset`.
        """
        return self.ptr + offset

    @always_inline
    fn __getitem__(self, idx: Int) raises -> Scalar[Self.dtype]:
        """Return the element at index `idx`.

        No bounds checking is performed.

        Args:
            idx: Element index.

        Returns:
            The scalar value at `idx`.
        """
        return self.ptr[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]) raises:
        """Set the element at index `idx` to `val`.

        No bounds checking is performed.

        Args:
            idx: Element index.
            val: Value to store.
        """
        self.ptr[idx] = val

    @always_inline
    fn load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector of `width` elements starting at `offset`.

        No bounds checking is performed.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            offset: Element index of the first lane.

        Returns:
            A SIMD vector of the requested width.
        """
        return self.ptr.load[width=width](offset)

    @always_inline
    fn store[width: Int](mut self, offset: Int, value: SIMD[Self.dtype, width]):
        """Store a SIMD vector of `width` elements starting at `offset`.

        No bounds checking is performed.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            offset: Element index of the first lane.
            value: The SIMD vector to write.
        """
        self.ptr.store[width=width](offset, value)

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Return the number of elements.

        Returns:
            `self.size`.
        """
        return self.size

    @always_inline
    fn __str__(self) -> String:
        """Return a human-readable summary of the container.

        Returns:
            A string of the form
            ``HostStorage(managed, size=N, refcount=R)`` or
            ``HostStorage(external, size=N)``.
        """
        if self.ownership == Ownership.External:
            return "HostStorage(external, size=" + String(self.size) + ")"
        return (
            "HostStorage(managed, size="
            + String(self.size)
            + ", refcount="
            + String(self.ref_count())
            + ")"
        )

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write a human-readable summary to `writer`.

        Parameters:
            W: The writer type.

        Args:
            writer: Destination writer.
        """
        if self.ownership == Ownership.External:
            writer.write("HostStorage(external, size=")
            writer.write(String(self.size))
            writer.write(")")
        else:
            writer.write("HostStorage(managed, size=")
            writer.write(String(self.size))
            writer.write(", refcount=")
            writer.write(String(self.ref_count()))
            writer.write(")")

    # ===----------------------------------------------------------------------===#
    # Reference Counting and Sharing
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn is_refcounted(ref self) -> Bool:
        """Return True if this container tracks a reference count.

        External containers and containers whose refcount pointer is null
        return False.

        Returns:
            Whether reference counting is active.
        """
        return (
            self._refcount != UnsafePointer[Atomic[DType.uint64], Self.origin]()
        )

    @always_inline
    fn ref_count(ref self) -> UInt64:
        """Return the current reference count, or 0 if not tracked.

        Returns:
            The atomic refcount value.
        """
        if not self.is_refcounted():
            return 0
        return self._refcount[].load[ordering = Consistency.MONOTONIC]()

    fn deep_copy(self) -> Self:
        """Create an independent managed copy of this container.

        The returned container has its own allocation and a refcount of 1,
        regardless of the source ownership mode.

        Returns:
            A new `HostStorage` that owns a copy of the data.
        """
        if self.size == 0:
            return HostStorage[Self.dtype]()

        var result = HostStorage[Self.dtype](self.size)
        memcpy(dest=result.ptr, src=self.ptr, count=self.size)
        return result^

    fn share(mut self) raises -> HostStorage[Self.dtype]:
        """Create a new handle that shares this container's data and refcount.

        The reference count is atomically incremented so both the original
        and the returned container keep the allocation alive.

        Returns:
            A new `HostStorage` pointing to the same buffer.

        Raises:
            Error: If the container is externally managed (no refcount).
        """
        if self.ownership == Ownership.External or not self.is_refcounted():
            raise Error(
                NumojoError(
                    category="memory",
                    message="Cannot share externally managed data",
                    location="HostStorage.share()",
                )
            )

        _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

        var result = HostStorage[Self.dtype](
            ptr=self.ptr,
            size=self.size,
            refcount=self._refcount,
            ownership=self.ownership,
        )

        return result^


# ===----------------------------------------------------------------------=== #
# DeviceStorage
# ===----------------------------------------------------------------------=== #


struct DeviceStorage[dtype: DType, device: Device](Copyable, Movable):
    """Device (GPU) backing storage for `AcceleratorDataContainer`.

    Wraps a `DeviceBuffer[dtype]` obtained from a `DeviceContext`.
    Copying a `DeviceStorage` copies the `DeviceBuffer` handle (the
    runtime may share or duplicate the underlying allocation depending on
    the GPU backend).

    Parameters:
        dtype: The element type stored in the buffer.
        device: The target GPU device descriptor.
    """

    var buffer: DeviceBuffer[Self.dtype]
    """The GPU-side data buffer."""

    var size: Int
    """Number of elements in the buffer."""

    # ===----------------------------------------------------------------------===#
    # Constructors
    # ===----------------------------------------------------------------------===#

    fn __init__(out self, size: Int) raises:
        """Allocate a new GPU buffer for `size` elements.

        Args:
            size: Number of elements to allocate.

        Raises:
            Error: If no GPU accelerator is available.
        """
        comptime assert is_accelerator_available[
            Self.device
        ](), "NuMojo: No GPU accelerator available."
        # TODO: Use a device-specific or cached DeviceContext instead of
        # the default one, so the correct backend is selected.
        self.buffer = DeviceContext().enqueue_create_buffer[Self.dtype](size)
        self.size = size

    fn __init__(
        out self,
        buffer: DeviceBuffer[Self.dtype],
        size: Int,
    ):
        """Wrap an existing `DeviceBuffer`.

        Args:
            buffer: An already-allocated device buffer.
            size: Number of elements accessible in `buffer`.
        """
        self.buffer = buffer
        self.size = size

    fn __copyinit__(out self, copy: Self):
        """Shallow-copy constructor.

        Copies the `DeviceBuffer` handle.  The GPU runtime determines
        whether the underlying memory is shared or duplicated.

        Args:
            copy: The source storage.
        """
        self.buffer = copy.buffer
        self.size = copy.size

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor.

        Transfers the buffer handle without copying.

        Args:
            take: The source storage (consumed).
        """
        self.buffer = take.buffer^
        self.size = take.size

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Return the number of elements.

        Returns:
            `self.size`.
        """
        return self.size

    @always_inline
    fn __str__(self) -> String:
        """Return a human-readable summary of the container.

        Returns:
            A string of the form
            ``DeviceStorage(device, size=N)``.
        """
        return (
            "DeviceStorage("
            + String(Self.device)
            + ", size="
            + String(self.size)
            + ")"
        )

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write a human-readable summary to `writer`.

        Parameters:
            W: The writer type.

        Args:
            writer: Destination writer.
        """
        writer.write(self.__str__())

    # ===----------------------------------------------------------------------===#
    # Access
    # ===----------------------------------------------------------------------===#

    fn get_buffer(ref self) -> ref[self.buffer] DeviceBuffer[Self.dtype]:
        """Return a reference to the underlying `DeviceBuffer`.

        Returns:
            A reference to `self.buffer`.
        """
        return self.buffer

    fn unsafe_ptr(ref self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Return the raw device pointer to the buffer's data.

        Returns:
            An `UnsafePointer` to the first element on the device.
        """
        return self.buffer.unsafe_ptr()


# ===----------------------------------------------------------------------=== #
# AcceleratorDataContainer
# ===----------------------------------------------------------------------=== #


struct AcceleratorDataContainer[dtype: DType, device: Device = Device.CPU](
    Copyable & Movable & Sized & Stringable & Writable
):
    """Unified, reference-counted storage for Host (CPU) or Device (GPU) data.

    At compile time the `device` parameter selects the backend:

    - **CPU** — delegates to `HostStorage` (atomic refcounted host memory).
    - **GPU** — delegates to `DeviceStorage` (device buffer handle).

    Only the field corresponding to the active backend is populated;
    the other remains `None`.

    Shallow copies (via `__copyinit__`) share the underlying allocation
    and increment the reference count.  Use `deep_copy()` for an
    independent owned copy.

    Parameters:
        dtype: The element type stored in the container.
        device: The execution device (default `Device.CPU`).
    """

    comptime origin = MutExternalOrigin
    """Memory origin for the container."""

    var host_storage: Optional[HostStorage[Self.dtype]]
    """Host (CPU) storage backend.  `None` for GPU containers."""

    var device_storage: Optional[DeviceStorage[Self.dtype, Self.device]]
    """Device (GPU) storage backend.  `None` for CPU containers."""

    var size: Int
    """Number of elements in the container."""

    # ===----------------------------------------------------------------------===#
    # Constructors and Destructor
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self, size: Int) raises:
        """Allocate storage for `size` elements on the target device.

        Args:
            size: Number of elements to allocate (must be non-negative).

        Raises:
            Error: If the requested GPU backend is unavailable, or the
                   device type is unrecognised.
        """
        if size < 0:
            abort(
                "AcceleratorDataContainer: __init__() size must be non-negative"
            )

        self.size = size

        comptime if Self.device.type == "cpu":
            self.host_storage = HostStorage[Self.dtype](size)
            self.device_storage = None
        elif Self.device.type == "gpu":
            if not is_accelerator_available[Self.device]():
                raise Error(
                    "\n Requested GPU device: "
                    + String(Self.device)
                    + " is not available. The available devices are: "
                    + Device.available_devices()
                )
            self.host_storage = None
            self.device_storage = DeviceStorage[Self.dtype, Self.device](size)
        else:
            raise Error("Unsupported device type: " + String(Self.device.type))

    @always_inline
    fn __init__(out self):
        """Create an empty container with no storage allocated."""
        self.host_storage = None
        self.device_storage = None
        self.size = 0

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        size: Int,
        copy: Bool = False,
    ) raises:
        """Create a CPU container from an existing host pointer.

        When `copy` is False the underlying `HostStorage` is external
        (non-owning).  When `copy` is True the data is deep-copied into
        a new managed allocation.

        Args:
            ptr: Pointer to an existing data buffer (must be non-null).
            size: Number of elements in the buffer (must be non-negative).
            copy: If True, deep-copy into owned storage.

        Constraints:
            Only valid for CPU devices.
        """
        comptime assert (
            Self.device.type == "cpu"
        ), "Pointer-based constructor is only valid for CPU devices"
        if size < 0:
            abort(
                "AcceleratorDataContainer: __init__() size must be non-negative"
            )
        if not ptr:
            abort("AcceleratorDataContainer: __init__() ptr must be non-null")

        self.size = size
        self.host_storage = HostStorage[Self.dtype](
            ptr.unsafe_origin_cast[HostStorage[Self.dtype].origin](),
            size,
            copy,
        )
        self.device_storage = None

    @always_inline
    fn __copyinit__(out self, copy: Self):
        """Shallow-copy constructor.

        Shares the underlying storage.  For CPU containers this increments
        the `HostStorage` atomic refcount; for GPU containers this copies
        the `DeviceStorage` handle (automatic reference counting).

        Args:
            copy: The source container.
        """
        self.host_storage = copy.host_storage
        self.device_storage = copy.device_storage
        self.size = copy.size

    @always_inline
    fn __moveinit__(out self, deinit take: Self):
        """Move constructor.

        Transfers all fields without touching reference counts.

        Args:
            take: The source container (consumed).
        """
        self.host_storage = take.host_storage^
        self.device_storage = take.device_storage^
        self.size = take.size

    # ===----------------------------------------------------------------------===#
    # Data Access (CPU)
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.dtype], MutExternalOrigin] where (
        Self.device.type == "cpu"
    ):
        """Return a pointer advanced by `offset` elements.

        Args:
            offset: Number of elements to advance.

        Returns:
            `self.host_storage.ptr + offset`.

        Constraints:
            CPU containers only.
        """
        return self.host_storage.unsafe_value().ptr + offset

    @always_inline
    fn __getitem__(
        self, idx: Int
    ) -> Scalar[Self.dtype] where Self.device.type == "cpu":
        """Return the element at index `idx`.

        No bounds checking is performed.

        Args:
            idx: Element index.

        Returns:
            The scalar value at `idx`.

        Constraints:
            CPU containers only.
        """
        return self.host_storage.unsafe_value().ptr[idx]

    @always_inline
    fn __setitem__(
        mut self, idx: Int, val: Scalar[Self.dtype]
    ) where Self.device.type == "cpu":
        """Set the element at index `idx` to `val`.

        No bounds checking is performed.

        Args:
            idx: Element index.
            val: Value to store.

        Constraints:
            CPU containers only.
        """
        self.host_storage.unsafe_value().ptr[idx] = val

    @always_inline
    fn load[
        width: Int
    ](self, offset: Int) -> SIMD[Self.dtype, width] where (
        Self.device.type == "cpu"
    ):
        """Load a SIMD vector of `width` elements starting at `offset`.

        No bounds checking is performed.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            offset: Element index of the first lane.

        Returns:
            A SIMD vector of the requested width.

        Constraints:
            CPU containers only.
        """
        return self.host_storage.unsafe_value().ptr.load[width=width](offset)

    @always_inline
    fn store[
        width: Int
    ](mut self, offset: Int, value: SIMD[Self.dtype, width]) where (
        Self.device.type == "cpu"
    ):
        """Store a SIMD vector of `width` elements starting at `offset`.

        No bounds checking is performed.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            offset: Element index of the first lane.
            value: The SIMD vector to write.

        Constraints:
            CPU containers only.
        """
        self.host_storage.unsafe_value().ptr.store[width=width](offset, value)

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Return the number of elements.

        Returns:
            `self.size`.
        """
        return self.size

    @always_inline
    fn __str__(self) -> String:
        """Return a human-readable summary of the container.

        Returns:
            A string indicating the device backend, size, and (for CPU)
            the underlying `HostStorage` representation.
        """

        comptime if Self.device.type == "cpu":
            if self.host_storage:
                return (
                    "AcceleratorDataContainer(cpu, "
                    + String(self.host_storage.unsafe_value())
                    + ")"
                )
            return "AcceleratorDataContainer(cpu, empty)"
        else:
            return (
                "AcceleratorDataContainer(gpu, size=" + String(self.size) + ")"
            )

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write a human-readable summary to `writer`.

        Parameters:
            W: The writer type.

        Args:
            writer: Destination writer.
        """
        writer.write(self.__str__())

    # ===----------------------------------------------------------------------===#
    # Reference Counting and Sharing
    # ===----------------------------------------------------------------------===#

    fn deep_copy(self) raises -> Self:
        """Create an independent managed copy of this container.

        For CPU containers the data is copied via `memcpy`.  For GPU
        containers the copy is enqueued on the device context.

        Returns:
            A new `AcceleratorDataContainer` that owns its own data.
        """
        if self.size == 0:
            return AcceleratorDataContainer[Self.dtype, Self.device]()

        comptime if Self.device.type == "cpu":
            var result = AcceleratorDataContainer[Self.dtype, Self.device](
                self.size
            )
            memcpy(
                dest=result.host_storage.unsafe_value().ptr,
                src=self.host_storage.unsafe_value().ptr,
                count=self.size,
            )
            return result^
        else:
            var result = AcceleratorDataContainer[Self.dtype, Self.device](
                self.size
            )
            var ctx = self.device_storage.unsafe_value().buffer.context()
            ctx.enqueue_copy(
                result.device_storage.unsafe_value().buffer,
                self.device_storage.unsafe_value().buffer,
            )
            return result^

    fn share(
        mut self,
    ) raises -> AcceleratorDataContainer[Self.dtype, Self.device]:
        """Create a new handle that shares this container's storage.

        For CPU containers the `HostStorage` refcount is atomically
        incremented.  For GPU containers the `DeviceStorage` handle is
        copied (the runtime manages device-side sharing).

        Returns:
            A new `AcceleratorDataContainer` backed by the same allocation.

        Raises:
            Error: If the active storage is missing or cannot be shared.
        """
        var result = AcceleratorDataContainer[Self.dtype, Self.device]()
        result.size = self.size

        comptime if Self.device.type == "cpu":
            var shared = self.host_storage.unsafe_value().share()
            result.host_storage = shared^
            result.device_storage = None
        elif Self.device.type == "gpu":
            result.device_storage = self.device_storage.copy()
            result.host_storage = None
        else:
            raise Error("Unsupported device type for sharing")

        return result^

    # ===----------------------------------------------------------------------===#
    # Device-Specific Access
    # ===----------------------------------------------------------------------===#

    @parameter
    fn is_cpu(self) -> Bool:
        """Return True if this container targets a CPU device."""
        return Self.device.type == "cpu"

    @parameter
    fn is_gpu(self) -> Bool:
        """Return True if this container targets a GPU device."""
        return Self.device.type == "gpu"

    @parameter
    fn is_cuda(self) -> Bool:
        """Return True if this container targets an NVIDIA CUDA device."""
        return Self.device == Device.CUDA

    @parameter
    fn is_rocm(self) -> Bool:
        """Return True if this container targets an AMD ROCm device."""
        return Self.device == Device.ROCM

    @parameter
    fn is_mps(self) -> Bool:
        """Return True if this container targets an Apple Metal device."""
        return Self.device == Device.MPS

    fn host_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin] where (
        Self.device == Device.CPU
    ):
        """Return the raw host pointer to the CPU allocation.

        Constraints:
            Only valid when `device` is `Device.CPU`.

        Returns:
            An `UnsafePointer` to the first element on the host.
        """
        return self.host_storage.unsafe_value().unsafe_ptr()

    fn device_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin] where (
        Self.device == Device.CUDA
        or Self.device == Device.ROCM
        or Self.device == Device.MPS
    ):
        """Return the raw device pointer to the GPU allocation.

        Constraints:
            Only valid for GPU devices (CUDA / ROCm / MPS).

        Returns:
            An `UnsafePointer` to the first element on the device.
        """
        return self.device_storage.unsafe_value().unsafe_ptr()

    fn host_buffer(
        self,
    ) -> HostStorage[Self.dtype] where Self.device == Device.CPU:
        """Return a shallow copy of the underlying `HostStorage`.

        The returned copy shares the same data pointer and refcount
        (the refcount is incremented).

        Constraints:
            Only valid for CPU containers.

        Returns:
            A copy of the `HostStorage`.
        """
        return self.host_storage.unsafe_value().copy()

    fn device_buffer(
        self,
    ) -> DeviceStorage[Self.dtype, Self.device] where (
        Self.device == Device.CUDA
        or Self.device == Device.ROCM
        or Self.device == Device.MPS
    ):
        """Return a shallow copy of the underlying `DeviceStorage`.

        Constraints:
            Only valid for GPU devices (CUDA / ROCm / MPS).

        Returns:
            A copy of the `DeviceStorage`.
        """
        return self.device_storage.unsafe_value().copy()
