# ===----------------------------------------------------------------------=== #
# NuMojo: DataContainer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Data Container (numojo.core.memory.data_container).
===================================================
Reference-counted memory container for array data.

Manages memory ownership and reference counting for contiguous data buffers
used by NDArray types.

Exports
-------
- `DataContainer`: Reference-counted container for array data.
- `Ownership`: Enum for managed vs external data ownership.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.atomic import (
    Atomic,
    fence,
    Ordering,
)
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.os import abort

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError


struct Ownership(ImplicitlyCopyable):
    """
    Enum indicating whether a DataContainer owns its data or views external data.

    - Managed: Container owns its data and uses reference counting.
    - External: Container views externally managed data and does not deallocate or refcount.
    """

    var value: Bool
    """Ownership status as a boolean."""

    comptime Managed = Ownership(True)
    """Container owns and manages its data."""

    comptime External = Ownership(False)
    """Container views externally managed data."""

    def __init__(out self, value: Bool):
        """
        Initialize Ownership with the given status.
        """
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        """
        Return True if both Ownership instances have the same status.
        """
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        """
        Return True if Ownership instances have different statuses.
        """
        return self.value != other.value

    def __xor__(self, other: Self) -> Bool:
        """
        Return True if Ownership statuses differ.
        """
        return self.value ^ other.value

    def __str__(self) -> String:
        """
        Return "Managed" or "External" based on ownership status.
        """
        if self == Ownership.Managed:
            return "Managed"
        else:
            return "External"

    def write_to[W: Writer](self, mut writer: W):
        """
        Write the ownership status as a string to the writer.
        """
        writer.write(self.__str__())


struct DataContainer[dtype: DType](Copyable & Sized & Writable):
    """
    Reference-counted container for a contiguous buffer of elements.

    DataContainer can either own its memory (managed) or provide a view into external data (external).
    Managed containers use reference counting for shared ownership. External containers do not manage or free memory.

    Copying a DataContainer with `.copy()` creates an independent owned copy with its own allocation.
    Use `.share()` to create a shared view that increments the reference count.

    Fields:
        ptr: Pointer to the data array.
        _refcount: Pointer to the atomic reference count (null for external).
        ownership: Ownership status (Managed or External).
        size: Number of elements in the data array.
    """

    comptime origin = MutUntrackedOrigin
    """Memory origin for the allocation."""

    var ptr: Pointer[Scalar[Self.dtype], Self.origin]
    """Pointer to the data array."""

    var _refcount: Pointer[Atomic[DType.uint64], Self.origin]
    """Pointer to the atomic reference count."""

    var ownership: Ownership
    """Ownership status of the container."""

    var size: Int
    """Number of elements in the data array."""

    # ===----------------------------------------------------------------------===#
    # Constructors and Destructor
    # ===----------------------------------------------------------------------===#
    @always_inline
    def __init__(out self):
        """
        Create an empty, managed DataContainer.
        """
        self.ptr = Pointer[Scalar[Self.dtype], Self.origin].unsafe_dangling()
        self._refcount = unsafe_alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed
        self.size = 0

    @always_inline
    def __init__(out self, size: Int):
        """
        Create a managed DataContainer with a buffer of `size` elements.

        Args:
            size: Number of elements to allocate (must be non-negative).
        """
        if size < 0:
            abort("DataContainer: __init__() size must be non-negative")

        self.size = size
        self._refcount = unsafe_alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed

        if size == 0:
            self.ptr = Pointer[
                Scalar[Self.dtype], Self.origin
            ].unsafe_dangling()
        else:
            self.ptr = unsafe_alloc[Scalar[Self.dtype]](size)

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], Self.origin],
        size: Int,
        copy: Bool = False,
    ):
        """
        Create a DataContainer from an existing buffer.

        If `copy` is True, the data is deep-copied into managed storage.
        If `copy` is False, the container is external and does not manage or
        free the memory.

        Args:
            ptr: Pointer to an existing data buffer (must be non-null).
            size: Number of elements in the buffer (must be non-negative).
            copy: If True, deep-copy into owned storage.
        """
        if size < 0:
            abort("DataContainer: __init__() size must be non-negative")
        self.size = size
        if copy:
            self._refcount = unsafe_alloc[Atomic[DType.uint64]](1)
            self._refcount[] = Atomic[DType.uint64](1)
            self.ptr = unsafe_alloc[Scalar[Self.dtype]](size)
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
        """Create a DataContainer that shares an existing buffer and refcount.

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
        """
        Deep copy constructor. Allocates new storage and copies all data.

        Args:
            copy: DataContainer to copy from.
        """
        self.size = copy.size
        self.ownership = Ownership.Managed
        self._refcount = unsafe_alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        if copy.size == 0:
            self.ptr = Pointer[
                Scalar[Self.dtype], Self.origin
            ].unsafe_dangling()
        else:
            self.ptr = unsafe_alloc[Scalar[Self.dtype]](copy.size)
            unsafe_memcpy(dest=self.ptr, src=copy.ptr, count=copy.size)

    @always_inline
    def __init__(out self, *, deinit move: Self):
        """
        Move constructor.

        Transfers ownership without changing the reference count.

        Args:
            move: DataContainer to move from.
        """
        self.ptr = move.ptr
        self._refcount = move._refcount
        self.ownership = move.ownership
        self.size = move.size

    @always_inline
    def __deinit__(deinit self):
        """
        Destructor.

        Decrements the reference count and frees memory if this is the last reference.
        """
        if self.ownership == Ownership.External:
            return

        if not self.is_refcounted():
            return

        if self._refcount[].fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return

        fence[ordering=Ordering.ACQUIRE]()
        if self.size > 0:
            self.ptr.unsafe_free()
        self._refcount.unsafe_free()

    # ===----------------------------------------------------------------------===#
    # Data Access Methods
    # ===----------------------------------------------------------------------===#
    @always_inline
    def get_ptr(
        ref self,
    ) -> ref[self.ptr] Pointer[Scalar[Self.dtype], Self.origin]:
        """
        Return a reference to the data pointer.

        Returns:
            Reference to the data pointer.
        """
        return self.ptr

    @always_inline
    def offset(self, offset: Int) -> Pointer[Scalar[Self.dtype], Self.origin]:
        """
        Return a pointer offset by the specified number of elements.

        Args:
            offset: Number of elements to offset from the start.

        Returns:
            Pointer to the element at the given offset.
        """
        return self.ptr.unsafe_offset(offset)

    @always_inline
    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """
        Return the element at the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The element at the given index.

        Notes:
            No bounds checking is performed. Caller must ensure index is valid.
        """
        return self.ptr[unsafe_offset=idx]

    @always_inline
    def __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]):
        """
        Set the element at the specified index to the given value.

        Args:
            idx: Index of the element to set.
            val: Value to assign.

        Notes:
            No bounds checking is performed. Caller must ensure index is valid.
        """
        self.ptr[unsafe_offset=idx] = val

    @always_inline
    def load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
        """
        Load a SIMD vector of the specified width from the given offset.

        Parameters:
            width: Number of elements in the SIMD vector.

        Args:
            offset: Index of the first element to load.

        Returns:
            SIMD vector of type `Self.dtype` and length `width` loaded from the specified offset.

        Notes:
            No bounds checking is performed. Caller must ensure there are enough elements from `offset`.
        """
        return self.ptr.unsafe_load[width=width](offset)

    @always_inline
    def store[
        width: Int = 1
    ](mut self, offset: Int, value: SIMD[Self.dtype, width]):
        """
        Store a SIMD vector of the specified width at the given offset.

        Parameters:
            width: Number of elements in the SIMD vector.

        Args:
            offset: Index at which to store the SIMD vector.
            value: SIMD vector to store.

        Notes:
            No bounds checking is performed. Caller must ensure there are enough elements from `offset`.
        """
        self.ptr.unsafe_store[width=width](offset, value)

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#
    @always_inline
    def __len__(self) -> Int:
        """
        Return the size of the container.

        Returns:
            The number of elements in the data array.
        """
        return self.size

    @always_inline
    def __str__(self) -> String:
        if self.ownership == Ownership.External:
            return "DataContainer(external, size=" + String(self.size) + ")"
        return (
            "DataContainer(managed, size="
            + String(self.size)
            + ", refcount="
            + String(self.ref_count())
            + ")"
        )

    @always_inline
    def write_to[W: Writer](self, mut writer: W):
        if self.ownership == Ownership.External:
            writer.write("DataContainer(external, size=")
            writer.write(String(self.size))
            writer.write(")")
        else:
            writer.write("DataContainer(managed, size=")
            writer.write(String(self.size))
            writer.write(", refcount=")
            writer.write(String(self.ref_count()))
            writer.write(")")

    # ===----------------------------------------------------------------------===#
    # Reference Counting and Sharing
    # ===----------------------------------------------------------------------===#
    @always_inline
    def is_refcounted(ref self) -> Bool:
        """
        Check if this container has refcounting enabled.

        Returns:
            True if refcounting is enabled, False otherwise.
        """
        return self.ownership == Ownership.Managed

    @always_inline
    def ref_count(ref self) -> UInt64:
        """
        Get the current reference count.

        Returns:
            The current reference count if refcounting is enabled, or 0 if not.
        """
        if not self.is_refcounted():
            return 0
        return self._refcount[].load[ordering=Ordering.RELAXED]()

    def share(self) raises -> DataContainer[Self.dtype]:
        """
        Create a shared view into this container.
        Increments the existing refcount for managed containers.

        Returns:
            A new DataContainer sharing the same data buffer, with refcount incremented if applicable.

        Raises:
            NumojoError: If the container is externally managed.
        """
        if self.ownership == Ownership.External or not self.is_refcounted():
            raise Error(
                NumojoError(
                    category="memory",
                    message="Cannot share externally managed data",
                    location="DataContainer.share()",
                )
            )

        _ = self._refcount[].fetch_add[ordering=Ordering.RELAXED](1)

        var result = DataContainer[Self.dtype](
            ptr=self.ptr,
            size=self.size,
            refcount=self._refcount,
            ownership=self.ownership,
        )

        return result^
