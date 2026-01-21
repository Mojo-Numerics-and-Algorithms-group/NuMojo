from memory import UnsafePointer
from os.atomic import Atomic, Consistency, fence
from sys import size_of


struct DataContainer[dtype: DType, origin: MutOrigin = MutExternalOrigin](
    ImplicitlyCopyable & Movable & Sized & Stringable & Writable
):
    """
    Reference-counted data container for matrix storage.

    Uses a single allocation with layout: [refcount: 8 bytes][data array]
    When shared, multiple DataContainers can point to the same allocation.
    The allocation is freed when the last reference is dropped.
    """

    var ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]
    var _refcount: UnsafePointer[Atomic[DType.uint64], Self.origin]
    var _alloc_start: UnsafePointer[UInt8, Self.origin]
    var size: Int

    @always_inline
    fn __init__(out self):
        """Initialize an empty container."""
        self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        self._refcount = UnsafePointer[Atomic[DType.uint64], Self.origin]()
        self._alloc_start = UnsafePointer[UInt8, Self.origin]()
        self.size = 0

    @always_inline
    fn __init__(out self, size: Int):
        """
        Allocate a new refcounted buffer of the given size.

        Memory layout: [Atomic refcount][data array]
        Initial refcount is 1.
        """
        debug_assert(size >= 0, "DataContainer: size must be >= 0")
        self.size = size

        if size == 0:
            self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
            self._refcount = UnsafePointer[Atomic[DType.uint64], Self.origin]()
            self._alloc_start = UnsafePointer[UInt8, Self.origin]()
            return

        var refcount_size = size_of[Atomic[DType.uint64]]()
        var data_size = size * size_of[Scalar[Self.dtype]]()
        var total_size = refcount_size + data_size

        var alloc_start = alloc[UInt8](total_size).unsafe_origin_cast[Self.origin]()
        var refcount_ptr = alloc_start.bitcast[Atomic[DType.uint64]]()
        refcount_ptr[] = Atomic[DType.uint64](1)

        var data_ptr = (alloc_start + refcount_size).bitcast[
            Scalar[Self.dtype]
        ]()

        self._alloc_start = alloc_start
        self._refcount = refcount_ptr
        self.ptr = data_ptr

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], Self.origin],
        refcount: UnsafePointer[Atomic[DType.uint64], Self.origin],
        alloc_start: UnsafePointer[UInt8, Self.origin],
        size: Int,
    ):
        """
        Create a shared view into an existing allocation.
        Increments the refcount.
        """
        self.ptr = ptr
        self._refcount = refcount
        self._alloc_start = alloc_start
        self.size = size

        if self._is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

    @always_inline
    fn _is_refcounted(self) -> Bool:
        """Check if this container has refcounting enabled."""
        return (
            self._refcount != UnsafePointer[Atomic[DType.uint64], Self.origin]()
        )

    @always_inline
    fn ref_count(self) -> UInt64:
        """Get the current reference count."""
        if not self._is_refcounted():
            return 1
        return self._refcount[].load[ordering = Consistency.MONOTONIC]()

    @always_inline
    fn get_ptr(self) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Get the data pointer."""
        return self.ptr

    @always_inline
    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Get a pointer offset from the start."""
        return self.ptr + offset

    @always_inline
    fn load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector from the given offset."""
        return self.ptr.load[width=width](offset)

    @always_inline
    fn store[width: Int](mut self, offset: Int, value: SIMD[Self.dtype, width]):
        """Store a SIMD vector at the given offset."""
        self.ptr.store[width=width](offset, value)

    @always_inline
    fn __len__(self) -> Int:
        """Return the size of the container."""
        return self.size

    @always_inline
    fn __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Get the element at the given index."""
        return self.ptr[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]):
        """Set the element at the given index."""
        self.ptr[idx] = val

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy constructor - increments refcount for shared containers."""
        self.ptr = other.ptr
        self._refcount = other._refcount
        self._alloc_start = other._alloc_start
        self.size = other.size

        if self._is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)

    @always_inline
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor - no refcount change."""
        self.ptr = other.ptr
        self._refcount = other._refcount
        self._alloc_start = other._alloc_start
        self.size = other.size

    @always_inline
    fn __del__(deinit self):
        """
        Destructor - decrements refcount and frees allocation if last reference.
        Uses release-acquire ordering for thread safety.
        """
        if not self._is_refcounted():
            return

        if self._refcount[].fetch_sub[ordering = Consistency.RELEASE](1) != 1:
            return

        fence[ordering = Consistency.ACQUIRE]()
        self._alloc_start.free()

    @always_inline
    fn share(self) -> Self:
        """
        Create a shared reference to this container's allocation.
        The returned container points to the same data and increments refcount.
        """
        debug_assert(
            self._is_refcounted(),
            "DataContainer.share(): requires a refcounted container",
        )

        return Self(
            ptr=self.ptr,
            refcount=self._refcount,
            alloc_start=self._alloc_start,
            size=self.size,
        )

    @always_inline
    fn share_with_offset(self, offset: Int) -> Self:
        """
        Create a shared reference with an offset pointer.
        Used for creating views into subregions of the data.
        """
        debug_assert(
            self._is_refcounted(),
            (
                "DataContainer.share_with_offset(): requires a refcounted"
                " container"
            ),
        )

        return Self(
            ptr=self.ptr + offset,
            refcount=self._refcount,
            alloc_start=self._alloc_start,
            size=self.size,
        )

    @always_inline
    fn __str__(self) -> String:
        if self._is_refcounted():
            return (
                "DataContainer(shared, size="
                + String(self.size)
                + ", refcount="
                + String(self.ref_count())
                + ", ptr="
                + String(self.ptr)
                + ")"
            )
        return (
            "DataContainer(untracked, size="
            + String(self.size)
            + ", ptr="
            + String(self.ptr)
            + ")"
        )

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())
