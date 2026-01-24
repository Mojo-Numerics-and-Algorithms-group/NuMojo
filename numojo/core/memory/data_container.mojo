from memory import UnsafePointer
from os.atomic import Atomic, Consistency, fence
from sys import size_of
from memory import memcpy
from os import abort


struct DataContainer[dtype: DType](
    ImplicitlyCopyable & Movable & Sized & Stringable & Writable
):
    """
    Reference-counted data container for matrix storage.

    Uses a single allocation with layout: [refcount: 8 bytes][data array]
    When shared, multiple DataContainers can point to the same allocation.
    The allocation is freed when the last reference is dropped.
    """

    comptime origin: MutOrigin = MutAnyOrigin
    """Memory origin for the allocation."""

    var ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]
    """Pointer to the data array."""
    var _refcount: UnsafePointer[Atomic[DType.uint64], Self.origin]
    """Pointer to the atomic reference count."""
    var size: Int
    """Number of elements in the data array."""
    var ext_origin: Bool
    """Whether the data pointer is externally managed."""

    @always_inline
    fn __init__(out self):
        """Initialize an empty container."""
        self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        self._refcount = UnsafePointer[Atomic[DType.uint64], Self.origin]()
        self.size = 0
        self.ext_origin = False

    @always_inline
    fn __init__(out self, size: Int, ext_origin: Bool = False):
        """
        Allocate a new refcounted buffer of the given size.

        Memory layout: [Atomic refcount][data array]
        Initial refcount is 1.
        """
        if size < 0:
            abort("DataContainer: __init__() size must be non-negative")

        self.size = size
        self._refcount = UnsafePointer[Atomic[DType.uint64], Self.origin]()
        self.ext_origin = ext_origin

        if size == 0:
            self.ptr = UnsafePointer[Scalar[Self.dtype], Self.origin]()
        else:
            if ext_origin:
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
        Create a shared view into an existing allocation.
        Increments the refcount.
        """
        self.size = size
        self._refcount = UnsafePointer[Atomic[DType.uint64], MutAnyOrigin]()
        if copy:
            self.ptr = alloc[Scalar[Self.dtype]](size)
            memcpy(dest=self.ptr, src=ptr, count=size)
            self.ext_origin = False
        else:
            self.ptr = ptr
            self.ext_origin = True

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy constructor - increments refcount for shared containers."""
        self.size = other.size
        self.ptr = other.ptr
        self._refcount = other._refcount
        self.ext_origin = other.ext_origin

        if self._is_refcounted():
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)
        else:
            if self.size > 0 and not self.ext_origin:
                self.ptr = alloc[Scalar[Self.dtype]](self.size)
                memcpy(dest=self.ptr, src=other.ptr, count=self.size)

    @always_inline
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor - no refcount change."""
        self.ptr = other.ptr
        self._refcount = other._refcount
        self.ext_origin = other.ext_origin
        self.size = other.size

    @always_inline
    fn __del__(deinit self):
        """
        Destructor - decrements refcount and frees allocation if last reference.
        """
        if self.size == 0 or self.ext_origin or not self.ptr.__bool__():
            print("NO FREE")
            return

        if self._is_refcounted():
            if (
                self._refcount[].fetch_sub[ordering = Consistency.RELEASE](1)
                != 1
            ):
                print("REFCOUNT DECR")
                return

            fence[ordering = Consistency.ACQUIRE]()
            var refcount_size = size_of[Atomic[DType.uint64]]()
            var alloc_start = self._refcount.bitcast[UInt8]()
            alloc_start.free()
            print("ALLOC FREE")
        else:
            self.ptr.free()
            print("PTR FREE")

    @always_inline
    fn get_ptr(
        ref self,
    ) -> ref [self.ptr] UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Get the data pointer."""
        return self.ptr

    @always_inline
    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Get a pointer offset from the start."""
        return self.ptr + offset

    @always_inline
    fn __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Get the element at the given index."""
        var norm_idx: Int = idx
        debug_assert(idx >= 0 and idx < self.size, "Index out of bounds")
        return self.ptr[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]):
        """Set the element at the given index."""
        self.ptr[idx] = val

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

    # fn __getitem(self, slice: Slice) -> DataContainer[Self.dtype]:
    #     """
    #     Get a sub-container for the given slice. Creates a copy of the data.
    #     """
    #     var start, stop, step = slice.indices(self.size)
    #     var slice_size = (stop - start + step - 1) // step

    #     if slice_size == 0
    #         return DataContainer[Self.dtype]()

    #     var res = DataContainer[Self.dtype](slice_size)

    @always_inline
    fn _is_refcounted(self) -> Bool:
        """Check if this container has refcounting enabled."""
        return (
            self._refcount != UnsafePointer[Atomic[DType.uint64], Self.origin]()
        )

    @always_inline
    fn ref_count(self) -> UInt64:
        """Get the current reference count."""
        return self._refcount[].load[ordering = Consistency.MONOTONIC]()

    @always_inline
    fn enable_views(mut self) raises:
        """
        Enables sharing of this container by initializing the refcount.
        """
        if self._is_refcounted() or self.size == 0:
            return

        if self.ext_origin:
            raise Error(
                "DataContainer.enable_views(): cannot enable views on"
                " externally managed data"
            )

        var refcount_size = size_of[Atomic[DType.uint64]]()
        var data_size = self.size * size_of[Scalar[Self.dtype]]()
        var total_size = refcount_size + data_size
        var alloc_start = alloc[UInt8](total_size)

        var new_refcount_ptr = alloc_start.bitcast[Atomic[DType.uint64]]()
        new_refcount_ptr[] = Atomic[DType.uint64](1)

        var new_data_ptr = (alloc_start + refcount_size).bitcast[
            Scalar[Self.dtype]
        ]()
        memcpy(dest=new_data_ptr, src=self.ptr, count=self.size)

        self.ptr.free()
        self.ptr = new_data_ptr
        self._refcount = new_refcount_ptr

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

    fn share_with_offset(ref self, offset: Int) -> DataContainer[Self.dtype]:
        """
        Create a shared view into this container starting at the given offset.
        Increments the refcount.
        """
        return DataContainer[Self.dtype](
            ptr=self.ptr + offset,
            size=self.size - offset,
            copy=False,
        )
