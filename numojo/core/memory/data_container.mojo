# ===----------------------------------------------------------------------=== #
# NuMojo: DataContainer
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""DataContainer (numojo.core.memory.data_container)

DataContainer is a reference-counted data container for NDArray and Matrix.
"""
from memory import UnsafePointer
from os.atomic import Atomic, Consistency, fence

from memory import memcpy
from os import abort

struct Ownership(ImplicitlyCopyable):
    """
    Ownership status for DataContainer. This is an enum encoded as a UInt8 for compact storage in the DataContainer struct.

    There are two ownership states:
        - Managed: The container manages its data and always uses reference counting for deallocation.
        - External: The container views externally managed data and does not perform deallocation or refcounting
    """

    var value: UInt8
    """The ownership status encoded as an unsigned 8-bit integer."""

    comptime Managed = Ownership(0)
    """Managed ownership means the container uses reference counting and frees data when the last reference is dropped."""

    comptime External = Ownership(1)
    """External ownership means the container views externally managed data and does not perform deallocation or refcounting."""

    fn __init__(out self, value: UInt8):
        """
        Initialize the Ownership with the given status.

        Args:
            value: The ownership status encoded as an unsigned 8-bit integer. Should be one of the predefined constants (Managed, External).
        """
        if value > 1:
            abort("Ownership: Invalid Ownership value")
        self.value = value

    fn __eq__(self, other: Ownership) -> Bool:
        """
        Check if two Ownership instances are equal based on their owner status.

        Args:
            other: Another Ownership instance to compare against.
        """
        return self.value == other.value

    fn __neq__(self, other: Ownership) -> Bool:
        """
        Check if two Ownership instances are not equal based on their owner status.

        Args:
            other: Another Ownership instance to compare against.
        """
        return self.value != other.value

    fn __str__(self) -> String:
        """
        Return a string representation of the Ownership status.

        Returns:
            A string representing the ownership status (Managed, External).
        """
        if self.value == Ownership.Managed.value:
            return "Managed"
        else:
            return "External"

    fn write_to[W: Writer](self, mut writer: W):
        """
        Write the string representation of the Ownership status to a writer.

        Args:
            writer: A writer to which the ownership status string will be written.
        """
        writer.write(self.__str__())


struct DataContainer[dtype: DType](
    Copyable & Movable & Sized & Stringable & Writable
):
    """
    A flexible, reference-counted data container.

    DataContainer can either manage its memory with reference counting or provide a view into externally
    managed data. It manages a contiguous buffer of `Scalar[Self.dtype]` elements, with ownership semantics
    controlled by the `ownership` field.

    Managed containers always allocate and use a refcount, so shared views can be created without a
    separate enable step. External containers never allocate a refcount and never free data.

    Copying a managed DataContainer increments the refcount; copying an external container preserves a
    non-owning view. Use `deep_copy()` to create an owned copy of any container.

    Fields:
        - ptr: Pointer to the data array.
        - _refcount: Pointer to the atomic reference count for managed containers (null for external).
        - ownership: Ownership status of the container (Managed, External).
        - size: Number of elements in the data array.
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

    @always_inline
    fn __init__(out self):
        """
        Initialize an empty container with no allocation and managed ownership.
        """
        self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.ownership = Ownership.Managed
        self.size = 0

    @always_inline
    fn __init__(out self, size: Int):
        """
        Allocate a managed buffer of `size` elements.

        Args:
            size: Number of elements to allocate.
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
        Create a view into an existing allocation.

        If `copy` is True, this deep-copies into managed storage.
        Otherwise, the container is marked as external and does not refcount.

        Args:
            ptr: Pointer to an existing data buffer.
            size: Number of elements in the buffer.
            copy: Whether to deep-copy into owned storage.
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
        Copy constructor - increments refcount for managed containers.

        Args:
            copy: The DataContainer to copy from.
        """
        self.size = copy.size
        self.ptr = copy.ptr
        self._refcount = copy._refcount
        self.ownership = copy.ownership

        if self.is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

    fn deep_copy(self) -> Self:
        """
        Create a deep copy of this DataContainer, regardless of refcounting or ownership.

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
        Move constructor - no refcount change.

        Args:
            take: The DataContainer to move from.
        """
        self.ptr = take.ptr
        self._refcount = take._refcount
        self.ownership = take.ownership
        self.size = take.size

    @always_inline
    fn __del__(deinit self):
        """
        Destructor - decrements refcount and frees allocation if last reference.
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

    @always_inline
    fn get_ptr(
        ref self,
    ) -> ref[self.ptr] UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """
        Get the data pointer.

        Returns:
            A reference to the data pointer.
        """
        return self.ptr

    @always_inline
    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """
        Get a pointer offset from the start.

        Args:
            offset: The element offset to apply to the pointer.
        """
        return self.ptr + offset

    @always_inline
    fn __getitem__(self, idx: Int) raises -> Scalar[Self.dtype]:
        """
        Get the element at the given index.

        Args:
            idx: The index of the element to retrieve. Supports negative indexing.

        Raises:
            Error: If the index is out of bounds.

        Returns:
            The element at the specified index.

        Notes:
            Caller must ensure that the index is valid.
            No bounds checking is performed in this method for performance reasons.
        """
        return self.ptr[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]) raises:
        """
        Set the element at the given index.

        Args:
            idx: The index of the element to set. Supports negative indexing.
            val: The value to set at the specified index.

        Raises:
            Error: If the index is out of bounds.

        Notes:
            Caller must ensure that the index is valid.
            No bounds checking is performed in this method for performance reasons.
        """
        self.ptr[idx] = val

    @always_inline
    fn load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
        """
        Load a SIMD vector from the given offset.

        Parameters:
            width: The width of the SIMD vector to load.

        Args:
            offset: The element offset from which to load the SIMD vector.

        Returns:
            A SIMD vector of the specified width loaded from the given offset.

        Notes:
            Caller must ensure that the offset is valid and that there are enough elements
            remaining in the container to load a full SIMD vector of the specified width.
            No bounds checking is performed in this method for performance reasons.
        """
        return self.ptr.load[width=width](offset)

    @always_inline
    fn store[width: Int](mut self, offset: Int, value: SIMD[Self.dtype, width]):
        """
        Store a SIMD vector at the given offset.

        Parameters:
            width: The width of the SIMD vector to store.

        Args:
            offset: The element offset at which to store the SIMD vector.
            value: The SIMD vector to store at the specified offset.

        Notes:
            Caller must ensure that the offset is valid and that there are enough elements
            remaining in the container to load a full SIMD vector of the specified width.
            No bounds checking is performed in this method for performance reasons.
        """
        self.ptr.store[width=width](offset, value)

    @always_inline
    fn __len__(self) -> Int:
        """
        Return the size of the container.

        Returns:
            The number of elements in the data array.
        """
        return self.size

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

    @always_inline
    fn __str__(self) -> String:
        if self.ownership == Ownership.External:
            return (
                "DataContainer(external, size="
                + String(self.size)
                + ")"
            )
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

        var result = DataContainer[Self.dtype]()
        result.size = self.size
        result.ptr = self.ptr
        result._refcount = self._refcount
        result.ownership = self.ownership

        if self.is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

        return result^
