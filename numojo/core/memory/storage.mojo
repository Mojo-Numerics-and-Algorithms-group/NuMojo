# ===----------------------------------------------------------------------=== #
# NuMojo: Storage
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Storage (numojo.core.memory.storage)
---------------------------------------
Backend storage containers for accelerator-aware data management.

This module provides three storage structs:

- `HostStorage`: Reference-counted host (CPU) memory container.
- `DeviceStorage`: Device (GPU) memory container wrapping a `DeviceBuffer`.
- `AcceleratorDataContainer`: Unified container that selects between
  `HostStorage` and `DeviceStorage` at compile time based on a `Device` parameter.
"""
from std.memory import UnsafePointer
from std.atomic import Atomic, Ordering, fence
from std.os import abort
from std.collections.optional import Optional
from std.sys.info import has_accelerator
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext

from numojo.core.accelerator import Device, DeviceHandle
from numojo.core.accelerator.device import is_accelerator_available
from numojo.core.memory.data_container import Ownership
from numojo.core.error import NumojoError


# ===----------------------------------------------------------------------=== #
# HostStorage
# ===----------------------------------------------------------------------=== #


struct HostStorage[dtype: DType](Copyable & Movable & Sized & Writable):
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

    comptime origin = MutUntrackedOrigin
    """Memory origin for the allocation."""

    var ptr: Pointer[Scalar[Self.dtype], Self.origin]
    """Pointer to the data array."""

    var _refcount: Pointer[Atomic[DType.uint64], Self.origin]
    """Pointer to the atomic reference count (null for external containers)."""

    var ownership: Ownership
    """Ownership status of the container (Managed or External)."""

    var size: Int
    """Number of elements in the data array."""

    # ===----------------------------------------------------------------------===#
    # Constructors and Destructor
    # ===----------------------------------------------------------------------===#

    # TODO: Replace UnsafePointer with Optional[UnsafePointer] for ptr.
    @always_inline
    def __init__(out self):
        """Create an empty managed container with size 0 and refcount 1."""
        self.ptr = Pointer[
            Scalar[Self.dtype], Self.origin
        ].unsafe_dangling()
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed
        self.size = 0

    @always_inline
    def __init__(out self, size: Int):
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
            self.ptr = Pointer[
                Scalar[Self.dtype], Self.origin
            ].unsafe_dangling()
        else:
            self.ptr = alloc[Scalar[Self.dtype]](size)

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], Self.origin],
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

        self.size = size
        if copy:
            self._refcount = alloc[Atomic[DType.uint64]](1)
            self._refcount[] = Atomic[DType.uint64](1)
            self.ptr = alloc[Scalar[Self.dtype]](size)
            unsafe_memcpy(dest=self.ptr, src=ptr, count=size)
            self.ownership = Ownership.Managed
        else:
            self._refcount = Pointer[
                Atomic[DType.uint64], Self.origin
            ].unsafe_dangling()
            self.ptr = ptr
            self.ownership = Ownership.External

    @always_inline
    def __init__(
        out self,
        *,
        ptr: Pointer[Scalar[Self.dtype], Self.origin],
        size: Int,
        refcount: Pointer[Atomic[DType.uint64], Self.origin],
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
    def __init__(out self, *, copy: Self):
        """Deep-copy constructor.

        Matches `DataContainer`: allocate owned storage and copy the data.
        Use `share()` for a shallow shared handle.

        Args:
            copy: The source container.
        """
        self.size = copy.size
        self.ownership = Ownership.Managed
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        if copy.size == 0:
            self.ptr = Pointer[
                Scalar[Self.dtype], Self.origin
            ].unsafe_dangling()
        else:
            self.ptr = alloc[Scalar[Self.dtype]](copy.size)
            unsafe_memcpy(dest=self.ptr, src=copy.ptr, count=copy.size)

    @always_inline
    def __init__(out self, *, deinit take: Self):
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
    def __del__(deinit self):
        """Destructor.

        For managed containers the reference count is atomically
        decremented.  If this was the last reference, the data buffer and
        the refcount allocation are freed.  External containers are left
        untouched.
        """
        if self.ownership == Ownership.External:
            return

        # I think this branch might be redundant, check it.
        if not self.is_refcounted():
            return

        if self._refcount[].fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return

        fence[ordering=Ordering.ACQUIRE]()
        if self.size > 0:
            self.ptr.unsafe_free()
        self._refcount.unsafe_free()

    # ===----------------------------------------------------------------------===#
    # Data Access
    # ===----------------------------------------------------------------------===#

    @always_inline
    def unsafe_ptr(
        ref self,
    ) -> ref[self.ptr] Pointer[Scalar[Self.dtype], Self.origin]:
        """Return a reference to the raw data pointer.

        Returns:
            A reference to `self.ptr`.
        """
        return self.ptr

    @always_inline
    def get_ptr(
        ref self,
    ) -> ref[self.ptr] Pointer[Scalar[Self.dtype], Self.origin]:
        """Return a reference to the raw data pointer.

        This mirrors `DataContainer.get_ptr()`.
        """
        return self.ptr

    @always_inline
    def offset(
        self, offset: Int
    ) -> Pointer[Scalar[Self.dtype], Self.origin]:
        """Return a pointer advanced by `offset` elements.

        Args:
            offset: Number of elements to advance.

        Returns:
            `self.ptr + offset`.
        """
        return self.ptr + offset

    @always_inline
    def __getitem__(self, idx: Int) raises -> Scalar[Self.dtype]:
        """Return the element at index `idx`.

        No bounds checking is performed.

        Args:
            idx: Element index.

        Returns:
            The scalar value at `idx`.
        """
        return self.ptr[unsafe_offset=idx]

    @always_inline
    def __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]) raises:
        """Set the element at index `idx` to `val`.

        No bounds checking is performed.

        Args:
            idx: Element index.
            val: Value to store.
        """
        self.ptr[unsafe_offset=idx] = val

    @always_inline
    def load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector of `width` elements starting at `offset`.

        No bounds checking is performed.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            offset: Element index of the first lane.

        Returns:
            A SIMD vector of the requested width.
        """
        return self.ptr.unsafe_load[width=width](offset)

    @always_inline
    def store[
        width: Int = 1
    ](mut self, offset: Int, value: SIMD[Self.dtype, width]):
        """Store a SIMD vector of `width` elements starting at `offset`.

        No bounds checking is performed.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            offset: Element index of the first lane.
            value: The SIMD vector to write.
        """
        self.ptr.unsafe_store[width=width](offset, value)

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#

    @always_inline
    def __len__(self) -> Int:
        """Return the number of elements.

        Returns:
            `self.size`.
        """
        return self.size

    @always_inline
    def __str__(self) -> String:
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
    def write_to[W: Writer](self, mut writer: W):
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
    def is_refcounted(ref self) -> Bool:
        """Return True if this container tracks a reference count.

        External containers and containers whose refcount pointer is null
        return False.

        Returns:
            Whether reference counting is active.
        """
        return self.ownership == Ownership.Managed

    @always_inline
    def ref_count(ref self) -> UInt64:
        """Return the current reference count, or 0 if not tracked.

        Returns:
            The atomic refcount value.
        """
        if not self.is_refcounted():
            return 0
        return self._refcount[].load[ordering=Ordering.RELAXED]()

    def share(self) raises -> HostStorage[Self.dtype]:
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

        _ = self._refcount[].fetch_add[ordering=Ordering.RELAXED](1)

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

    var handle: DeviceHandle[Self.device]
    """Runtime handle that owns the device context used for this storage."""

    var size: Int
    """Number of elements in the buffer."""

    # ===----------------------------------------------------------------------===#
    # Constructors
    # ===----------------------------------------------------------------------===#

    def __init__(out self, size: Int) raises:
        """Allocate a new GPU buffer for `size` elements.

        Args:
            size: Number of elements to allocate.

        Raises:
            Error: If no GPU accelerator is available.
        """
        comptime assert is_accelerator_available[
            Self.device
        ](), "NuMojo: No GPU accelerator available."
        self.handle = DeviceHandle[Self.device]()
        var context = self.handle.device_context()
        self.buffer = context.enqueue_create_buffer[Self.dtype](size)
        self.size = size

    def __init__(
        out self,
        buffer: DeviceBuffer[Self.dtype],
        size: Int,
    ) raises:
        """Wrap an existing `DeviceBuffer`.

        Args:
            buffer: An already-allocated device buffer.
            size: Number of elements accessible in `buffer`.
        """
        var context = buffer.context()
        self.handle = DeviceHandle[Self.device](context^)
        self.buffer = buffer
        self.size = size

    def __init__(out self, *, copy: Self):
        """Deep-copy constructor.

        Allocates a new buffer on the same device context and copies all data.
        Use `share()` for a shallow shared handle.

        Args:
            copy: The source storage.
        """
        self.handle = copy.handle.copy()
        try:
            var context = self.handle.device_context()
            self.buffer = context.enqueue_create_buffer[Self.dtype](copy.size)
            copy.buffer.enqueue_copy_to(self.buffer)
            context.synchronize()
        except e:
            abort("DeviceStorage: deep copy failed: " + String(e))
        self.size = copy.size

    def __init__(out self, *, deinit take: Self):
        """Move constructor.

        Transfers the buffer handle without copying.

        Args:
            take: The source storage (consumed).
        """
        self.handle = take.handle^
        self.buffer = take.buffer^
        self.size = take.size

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#

    @always_inline
    def __len__(self) -> Int:
        """Return the number of elements.

        Returns:
            `self.size`.
        """
        return self.size

    @always_inline
    def __str__(self) -> String:
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
    def write_to[W: Writer](self, mut writer: W):
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

    def get_buffer(ref self) -> ref[self.buffer] DeviceBuffer[Self.dtype]:
        """Return a reference to the underlying `DeviceBuffer`.

        Returns:
            A reference to `self.buffer`.
        """
        return self.buffer

    def unsafe_ptr(ref self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Return the raw device pointer to the buffer's data.

        Returns:
            An `UnsafePointer` to the first element on the device.
        """
        return self.buffer.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()

    def share(self) raises -> DeviceStorage[Self.dtype, Self.device]:
        """Create a shallow handle sharing this device buffer."""
        var shared_buffer = self.buffer.copy()
        return DeviceStorage[Self.dtype, Self.device](shared_buffer^, self.size)


# ===----------------------------------------------------------------------=== #
# AcceleratorDataContainer
# ===----------------------------------------------------------------------=== #


struct AcceleratorDataContainer[dtype: DType, device: Device = Device.CPU](
    Copyable & Movable & Sized & Writable
):
    """Unified, reference-counted storage for Host (CPU) or Device (GPU) data.

    At compile time the `device` parameter selects the backend:

    - **CPU** — delegates to `HostStorage` (atomic refcounted host memory).
    - **GPU** — delegates to `DeviceStorage` (device buffer handle).

    Only the field corresponding to the active backend is populated;
    the other remains `None`.

    Shallow copies (via `__copyinit__`) share the underlying allocation
    and increment the reference count.  Use `.share()` for an explicit shared handle.

    Parameters:
        dtype: The element type stored in the container.
        device: The execution device (default `Device.CPU`).
    """

    comptime origin = MutUntrackedOrigin
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
    def __init__(out self, size: Int) raises:
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
    def __init__(out self):
        """Create an empty container with no storage allocated."""
        self.host_storage = None
        self.device_storage = None
        self.size = 0

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
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

        self.size = size
        self.host_storage = HostStorage[Self.dtype](
            ptr.unsafe_origin_cast[HostStorage[Self.dtype].origin](),
            size,
            copy,
        )
        self.device_storage = None

    @always_inline
    def __init__(
        out self, var host_storage: HostStorage[Self.dtype]
    ) where Self.device.type == "cpu":
        """Create a CPU container from an existing `HostStorage` handle."""
        self.size = host_storage.size
        self.host_storage = host_storage^
        self.device_storage = None

    @always_inline
    def __init__(
        out self, var device_storage: DeviceStorage[Self.dtype, Self.device]
    ) where Self.device.type == "gpu":
        """Create a GPU container from an existing `DeviceStorage` handle."""
        self.size = device_storage.size
        self.host_storage = None
        self.device_storage = device_storage^

    @always_inline
    def __init__(out self, *, copy: Self):
        """Deep-copy constructor.

        Copies the active backend container. Use `share()` for a shallow
        shared handle.

        Args:
            copy: The source container.
        """
        self.host_storage = copy.host_storage.copy()
        self.device_storage = copy.device_storage.copy()
        self.size = copy.size

    @always_inline
    def __init__(out self, *, deinit take: Self):
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
    def offset(
        self, offset: Int
    ) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin] where (
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
    def __getitem__(
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
        return self.host_storage.unsafe_value().ptr[unsafe_offset=idx]

    @always_inline
    def __setitem__(
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
        self.host_storage.unsafe_value().ptr[unsafe_offset=idx] = val

    @always_inline
    def load[
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
        return self.host_storage.unsafe_value().ptr.unsafe_load[width=width](offset)

    @always_inline
    def store[
        width: Int = 1
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
        self.host_storage.unsafe_value().ptr.unsafe_store[width=width](offset, value)

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#

    @always_inline
    def __len__(self) -> Int:
        """Return the number of elements.

        Returns:
            `self.size`.
        """
        return self.size

    @always_inline
    def __str__(self) -> String:
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
    def write_to[W: Writer](self, mut writer: W):
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

    def share(self) raises -> AcceleratorDataContainer[Self.dtype, Self.device]:
        """Create a new handle that shares this container's storage.

        For CPU containers the `HostStorage` refcount is atomically
        incremented.  For GPU containers the `DeviceStorage` handle is
        copied (the runtime manages device-side sharing).

        Returns:
            A new `AcceleratorDataContainer` backed by the same allocation.

        Raises:
            Error: If the active storage is missing or cannot be shared.
        """
        comptime if Self.device.type == "cpu":
            var shared = self.host_storage.unsafe_value().share()
            return AcceleratorDataContainer[Self.dtype, Self.device](shared^)
        elif Self.device.type == "gpu":
            var shared = self.device_storage.unsafe_value().share()
            return AcceleratorDataContainer[Self.dtype, Self.device](shared^)
        else:
            raise Error("Unsupported device type for sharing")

    # ===----------------------------------------------------------------------===#
    # Device-Specific Access
    # ===----------------------------------------------------------------------===#

    @parameter
    def is_cpu(self) -> Bool:
        """Return True if this container targets a CPU device."""
        return Self.device.type == "cpu"

    @parameter
    def is_gpu(self) -> Bool:
        """Return True if this container targets a GPU device."""
        return Self.device.type == "gpu"

    @parameter
    def is_cuda(self) -> Bool:
        """Return True if this container targets an NVIDIA CUDA device."""
        return Self.device == Device.CUDA

    @parameter
    def is_rocm(self) -> Bool:
        """Return True if this container targets an AMD ROCm device."""
        return Self.device == Device.ROCM

    @parameter
    def is_mps(self) -> Bool:
        """Return True if this container targets an Apple Metal device."""
        return Self.device == Device.MPS

    def host_ptr(
        self,
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin] where (
        Self.device == Device.CPU
    ):
        """Return the raw host pointer to the CPU allocation.

        Constraints:
            Only valid when `device` is `Device.CPU`.

        Returns:
            An `UnsafePointer` to the first element on the host.
        """
        return self.host_storage.unsafe_value().unsafe_ptr().as_unsafe_any_origin()

    def device_ptr(
        self,
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin] where (
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

    def host_buffer(
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

    def device_buffer(
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
