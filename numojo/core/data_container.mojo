# ===----------------------------------------------------------------------=== #
# Define `DataContainer` type
#
# TODO: fields in traits are not supported yet by Mojo
# Currently use `get_ptr()` to get pointer, in future, use `ptr` directly.
# var ptr: LegacyUnsafePointer[Scalar[dtype]]
# ===----------------------------------------------------------------------===

from memory import UnsafePointer, LegacyUnsafePointer


# temporary DataContainer to support transition from LegacyUnsafePointer to UnsafePointer.
struct DataContainerNew[dtype: DType, origin: MutOrigin](ImplicitlyCopyable):
    """
    DataContainer is managing a contiguous block of memory containing elements of type `Scalar[dtype]`, using an `UnsafePointer` with a specified mutability origin. It provides basic memory management and pointer access for low-level array operations.

    Type Parameters:
        dtype: The data type of the elements stored in the container.
        origin: The mutability origin for the pointer, controlling comptimeing and mutation semantics.
    """

    var ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]

    fn __init__(out self, size: Int):
        """
        Allocate given space on memory.
        The bytes allocated is `size` * `byte size of dtype`.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as True.
        The memory should be freed by `__del__`.
        """
        self.ptr: UnsafePointer[Scalar[Self.dtype], Self.origin] = alloc[Scalar[Self.dtype]](
            size
        ).unsafe_origin_cast[Self.origin]()

    fn __init__(out self, ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]):
        """
        Do not use this if you know what it means.
        If the pointer is associated with another array, it might cause
        dangling pointer problem.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as False.
        The memory should not be freed by `__del__`.
        """
        self.ptr = ptr

    fn __moveinit__(out self, deinit other: Self):
        """
        Move-initializes this DataContainerNew from another instance.

        Transfers ownership of the pointer from `other` to `self`.
        After this operation, `other` should not be used.
        """
        self.ptr = other.ptr

    fn get_ptr(
        self,
    ) -> ref [origin_of(self.ptr)] UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """
        Returns the internal pointer to the data buffer.

        Returns:
            UnsafePointer[Scalar[dtype], origin]: The pointer to the underlying data.
        """
        return self.ptr

    fn __str__(self) -> String:
        """
        Returns a string representation of the DataContainerNew.

        Returns:
            String: A string describing the container and its pointer.
        """
        return "DatContainer with ptr: " + String(self.ptr)

    fn __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """
        Gets the value at the specified index in the data buffer.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            Scalar[dtype]: The value at the given index.
        """
        return self.ptr[idx]

    fn __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]):
        """
        Sets the value at the specified index in the data buffer.

        Args:
            idx: Index of the element to set.
            val: Value to assign.
        """
        self.ptr[idx] = val

    fn offset(self, offset: Int) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """
        Returns a pointer offset by the given number of elements.

        Args:
            offset: Number of elements to offset the pointer.

        Returns:
            UnsafePointer[Scalar[dtype], origin]: The offset pointer.
        """
        return self.ptr.offset(offset)

    fn load[width: Int](self, offset: Int) -> SIMD[Self.dtype, width]:
        """
        Loads a value from the data buffer at the specified offset.

        Args:
            offset: Offset from the start of the buffer.

        Returns:
            Scalar[dtype]: The loaded value.
        """
        return self.ptr.load[width=width](offset)

    fn store[width: Int](mut self, offset: Int, value: SIMD[Self.dtype, width]):
        """
        Stores a value into the data buffer at the specified offset.

        Args:
            offset: Offset from the start of the buffer.
            value: Value to store.
        """
        self.ptr.store[width=width](offset, value)


struct DataContainer[dtype: DType](ImplicitlyCopyable):
    var ptr: LegacyUnsafePointer[Scalar[Self.dtype], origin=MutOrigin.external]

    fn __init__(out self, size: Int):
        """
        Allocate given space on memory.
        The bytes allocated is `size` * `byte size of dtype`.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as True.
        The memory should be freed by `__del__`.
        """
        self.ptr = LegacyUnsafePointer[Scalar[Self.dtype], origin=MutOrigin.external]().alloc(size)

    fn __init__(out self, ptr: LegacyUnsafePointer[Scalar[Self.dtype], origin=MutOrigin.external]):
        """
        Do not use this if you know what it means.
        If the pointer is associated with another array, it might cause
        dangling pointer problem.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as False.
        The memory should not be freed by `__del__`.
        """
        self.ptr = ptr

    fn __moveinit__(out self, deinit other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> ref [self.ptr] LegacyUnsafePointer[Scalar[Self.dtype], origin=MutOrigin.external]:
        return self.ptr
