# ===----------------------------------------------------------------------=== #
# NuMojo: DataContainer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""DataContainer (numojo.core.memory.data_container)

A reference-counted container for contiguous data buffers, used for NDArray and Matrix.

DataContainer manages memory ownership and reference counting for shared or external data.
"""
from memory import UnsafePointer
from os.atomic import Atomic, Consistency, fence

from memory import memcpy
from os import abort

from numojo.core.error import NumojoError


struct Ownership(ImplicitlyCopyable):
    """
    Enum indicating whether a DataContainer owns its data or views external data.

    - Managed: Container owns its data and uses reference counting.
    - External: Container views externally managed data and does not deallocate or refcount.
    """

    var value: __mlir_type.i1
    """Ownership status as a boolean."""

    comptime Managed = Ownership(True._mlir_value)
    """Container owns and manages its data."""

    comptime External = Ownership(False._mlir_value)
    """Container views externally managed data."""

    fn __init__(out self, value: __mlir_type.i1):
        """
        Initialize Ownership with the given status.
        """
        self.value = value

    fn __eq__(self, other: Self) -> Bool:
        """
        Return True if both Ownership instances have the same status.
        """
        return ~(self != other)

    fn __ne__(self, other: Self) -> Bool:
        """
        Return True if Ownership instances have different statuses.
        """
        return self ^ other

    fn __xor__(self, other: Self) -> Bool:
        """
        Return True if Ownership statuses differ.
        """
        return __mlir_op.`pop.xor`(self.value, other.value)

    fn __str__(self) -> String:
        """
        Return "Managed" or "External" based on ownership status.
        """
        if self == Ownership.Managed:
            return "Managed"
        else:
            return "External"

    fn write_to[W: Writer](self, mut writer: W):
        """
        Write the ownership status as a string to the writer.
        """
        writer.write(self.__str__())


struct DataContainer[dtype: DType](
    Copyable & Movable & Sized & Stringable & Writable
):
    """
    Reference-counted container for a contiguous buffer of elements.

    DataContainer can either own its memory (managed) or provide a view into external data (external).
    Managed containers use reference counting for shared ownership. External containers do not manage or free memory.

    Copying a managed DataContainer with `.copy()` increments the reference count that still points to the same data.
    Copying an external container creates another non-owning view.
    Use `deep_copy()` to create an owned instance.

    Fields:
        ptr: Pointer to the data array.
        _refcount: Pointer to the atomic reference count (null for external).
        ownership: Ownership status (Managed or External).
        size: Number of elements in the data array.
    """

    comptime origin = MutExternalOrigin
    """Memory origin for the allocation."""

    var ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]
    """Pointer to the data array."""

    var _refcount: UnsafePointer[Atomic[DType.uint64], Self.origin]
    """Pointer to the atomic reference count."""

    var ownership: Ownership
    """Ownership status of the container."""

    var size: Int
    """Number of elements in the data array."""

    # ===----------------------------------------------------------------------===#
    # Constructors and Destructor
    # ===----------------------------------------------------------------------===#
    @always_inline
    fn __init__(out self):
        """
        Create an empty, managed DataContainer.
        """
        self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed
        self.size = 0

    @always_inline
    fn __init__(out self, size: Int):
        """
        Create a managed DataContainer with a buffer of `size` elements.

        Args:
            size: Number of elements to allocate (must be non-negative).
        """
        if size < 0:
            abort("DataContainer: __init__() size must be non-negative")

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
        if not ptr:
            abort("DataContainer: __init__() ptr must be non-null")
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
    fn __copyinit__(out self, copy: Self):
        """
        Copy constructor.

        Increments the reference count for managed containers.

        Args:
            copy: DataContainer to copy from.
        """
        self.size = copy.size
        self.ptr = copy.ptr
        self._refcount = copy._refcount
        self.ownership = copy.ownership

        if self.is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

    fn deep_copy(self) -> Self:
        """
        Create a deep copy of the DataContainer.

        Returns:
            A new DataContainer with its own copy of the data.
        """
        if self.size == 0:
            return DataContainer[Self.dtype]()

        var result = DataContainer[Self.dtype](self.size)
        memcpy(dest=result.ptr, src=self.ptr, count=self.size)
        return result^

    @always_inline
    fn __moveinit__(out self, deinit take: Self):
        """
        Move constructor.

        Transfers ownership without changing the reference count.

        Args:
            take: DataContainer to move from.
        """
        self.ptr = take.ptr
        self._refcount = take._refcount
        self.ownership = take.ownership
        self.size = take.size

    @always_inline
    fn __del__(deinit self):
        """
        Destructor.

        Decrements the reference count and frees memory if this is the last reference.
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
    # Data Access Methods
    # ===----------------------------------------------------------------------===#
    @always_inline
    fn get_ptr(
        ref self,
    ) -> ref[self.ptr] UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """
        Return a reference to the data pointer.

        Returns:
            Reference to the data pointer.
        """
        return self.ptr

    @always_inline
    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """
        Return a pointer offset by the specified number of elements.

        Args:
            offset: Number of elements to offset from the start.

        Returns:
            Pointer to the element at the given offset.
        """
        return self.ptr + offset

    @always_inline
    fn __getitem__(self, idx: Int) raises -> Scalar[Self.dtype]:
        """
        Return the element at the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The element at the given index.

        Notes:
            No bounds checking is performed. Caller must ensure index is valid.
        """
        return self.ptr[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]) raises:
        """
        Set the element at the specified index to the given value.

        Args:
            idx: Index of the element to set.
            val: Value to assign.

        Notes:
            No bounds checking is performed. Caller must ensure index is valid.
        """
        self.ptr[idx] = val

    @always_inline
    fn load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
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
        return self.ptr.load[width=width](offset)

    @always_inline
    fn store[width: Int](mut self, offset: Int, value: SIMD[Self.dtype, width]):
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
        self.ptr.store[width=width](offset, value)

    # ===----------------------------------------------------------------------===#
    # Trait Implementations
    # ===----------------------------------------------------------------------===#
    @always_inline
    fn __len__(self) -> Int:
        """
        Return the size of the container.

        Returns:
            The number of elements in the data array.
        """
        return self.size

    @always_inline
    fn __str__(self) -> String:
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
    fn write_to[W: Writer](self, mut writer: W):
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
    fn is_refcounted(ref self) -> Bool:
        """
        Check if this container has refcounting enabled.

        Returns:
            True if refcounting is enabled, False otherwise.
        """
        return (
            self._refcount != UnsafePointer[Atomic[DType.uint64], Self.origin]()
        )

    @always_inline
    fn ref_count(ref self) -> UInt64:
        """
        Get the current reference count.

        Returns:
            The current reference count if refcounting is enabled, or 0 if not.
        """
        if not self.is_refcounted():
            return 0
        return self._refcount[].load[ordering = Consistency.MONOTONIC]()

    fn share(mut self) raises -> DataContainer[Self.dtype]:
        """
        Create a shared view into this container.
        Increments the existing refcount for managed containers.

        Returns:
            A new DataContainer sharing the same data buffer, with refcount incremented if applicable.

        Raises:
            Error: If the container is externally managed.
        """
        if self.ownership == Ownership.External or not self.is_refcounted():
            raise Error(
                NumojoError(
                    category="memory",
                    message="Cannot share externally managed data",
                    location="DataContainer.share()",
                )
            )

        _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

        # Build the shared handle without allocating a new refcount.
        # The ptr-based ctor creates an External view; we then override
        # the fields to share the managed refcount.
        # Note that when copy is False, the ptr is null and no alloc occurs.
        var result = DataContainer[Self.dtype](self.ptr, self.size, copy=False)
        result._refcount = self._refcount
        result.ownership = self.ownership

        return result^
