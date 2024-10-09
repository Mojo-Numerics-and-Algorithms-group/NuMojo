from utils import Variant
from builtin.type_aliases import AnyLifetime

@value
struct _IdxIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for idx.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying idx data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Idx
    var length: Int

    fn __init__(
        inout self,
        array: Idx,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) raises -> Scalar[DType.index]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.array.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.array.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index


struct Idx(CollectionElement, Formattable):
    alias dtype: DType = DType.index
    alias width = simdwidthof[Self.dtype]()
    var storage: UnsafePointer[Scalar[Self.dtype]]
    var len: Int

    @always_inline("nodebug")
    fn __init__(inout self, owned *args: Scalar[Self.dtype]):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(args.__len__())
        self.len = args.__len__()
        for i in range(args.__len__()):
            self.storage[i] = args[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, owned args: Variant[List[Int], VariadicList[Int]]
    ) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        if args.isa[List[Int]]():
            self.len = args[List[Int]].__len__()
            self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(self.len)
            for i in range(self.len):
                self.storage[i] = args[List[Int]][i]
        elif args.isa[VariadicList[Int]]():
            self.len = args[VariadicList[Int]].__len__()
            self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(self.len)
            for i in range(self.len):
                self.storage[i] = args[VariadicList[Int]][i]
        else:
            raise Error("Invalid type")

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(
            other.__len__()
        )
        self.len = other.len
        for i in range(other.__len__()):
            self.storage[i] = other[i]

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Self):
        """Move construct the tuple.

        Args:
            other: The tuple to move.
        """
        self.storage = other.storage
        self.len = other.len

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.len

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        """Get the value at the specified index.

        Args:
            index: The index of the value to get.

        Returns:
            The value at the specified index.
        """
        return int(self.storage[index])

    @always_inline("nodebug")
    fn __setitem__(self, index: Int, val: Int):
        """Set the value at the specified index.

        Args:
            index: The index of the value to set.
            val: The value to set.
        """
        self.storage[index] = val

    fn __iter__(self) raises -> _IdxIter[__lifetime_of(self)]:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _IdxIter[__lifetime_of(self)](
            array=self,
            length=self.len,
        )

    fn format_to(self, inout writer: Formatter):
        writer.write("Idx: " + self.str() + "\n" + "Length: " + str(self.len))

    fn str(self) -> String:
        var result: String = "["
        for i in range(self.len):
            result += str(self.storage[i])
            if i < self.len - 1:
                result += ", "
        result += "]"
        return result

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[Self.dtype, width]:
        return self.storage.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[Self.dtype, width]) raises:
        self.storage.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> SIMD[Self.dtype, width]:
        return self.storage.load[width=width](index)

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[Self.dtype, width]):
        self.storage.store[width=width](index, val)

@register_passable("trivial")
struct NDArrayShape[dtype: DType = DType.int32](Stringable, Formattable):
    """Implements the NDArrayShape."""

    # Fields
    var ndsize: Int
    """Total no of elements in the corresponding array."""
    var ndshape: UnsafePointer[Scalar[dtype]]
    """Shape of the corresponding array."""
    var ndlen: Int
    """Length of ndshape."""

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.
        """
        self.ndsize = 1
        self.ndlen = len(shape)
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            self.ndsize *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int, size: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions and a specified size.

        Args:
            shape: Variable number of integers representing the shape dimensions.
            size: The total number of elements in the array.
        """
        self.ndsize = size
        self.ndlen = len(shape)
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self.ndsize = 1
        self.ndlen = len(shape)
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            self.ndsize *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self.ndsize = (
            size  # maybe I should add a check here to make sure it matches
        )
        self.ndlen = len(shape)
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self.ndsize = 1
        self.ndlen = len(shape)
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            self.ndsize *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self.ndsize = (
            size  # maybe I should add a check here to make sure it matches
        )
        self.ndlen = len(shape)
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(inout self, shape: NDArrayShape) raises:
        """
        Initializes the NDArrayShape with another NDArrayShape.

        Args:
            shape: Another NDArrayShape to initialize from.
        """
        self.ndsize = shape.ndsize
        self.ndlen = shape.ndlen
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(shape.ndlen)
        memset_zero(self.ndshape, shape.ndlen)
        for i in range(shape.ndlen):
            self.ndshape[i] = shape[i]

    fn __copy__(inout self, other: Self):
        """
        Copy from other into self.
        """
        self.ndsize = other.ndsize
        self.ndlen = other.ndlen
        self.ndshape = UnsafePointer[Scalar[dtype]]().alloc(other.ndlen)
        memcpy(self.ndshape, other.ndshape, other.ndlen)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Get shape at specified index.
        """
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            return self.ndshape[index].__int__()
        else:
            return self.ndshape[self.ndlen + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        """
        Set shape at specified index.
        """
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            self.ndshape[index] = val
        else:
            self.ndshape[self.ndlen + index] = val

    @always_inline("nodebug")
    fn size(self) -> Int:
        """
        Get Size of array described by arrayshape.
        """
        return self.ndsize

    @always_inline("nodebug")
    fn len(self) -> Int:
        """
        Get number of dimensions of the array described by arrayshape.
        """
        return self.ndlen

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Return a string of the shape of the array described by arrayshape.
        """
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        var result: String = "Shape: ["
        for i in range(self.ndlen):
            if i == self.ndlen - 1:
                result += self.ndshape[i].__str__()
            else:
                result += self.ndshape[i].__str__() + ", "
        result = result + "]"
        writer.write(result)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        """
        Check if two arrayshapes have identical dimensions.
        """
        for i in range(self.ndlen):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        """
        Check if two arrayshapes don't have identical dimensions.
        """
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        """
        Check if any of the dimensions are equal to a value.
        """
        for i in range(self.ndlen):
            if self[i] == val:
                return True
        return False

    # can be used for vectorized index calculation
    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        """
        SIMD load dimensional information.
        """
        # if index >= self.ndlen:
        # raise Error("Index out of bound")
        return self.ndshape.load[width=width](index)

    # can be used for vectorized index retrieval
    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        """
        SIMD store dimensional information.
        """
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        self.ndshape.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_int(self, index: Int) -> Int:
        """
        SIMD load dimensional information.
        """
        return self.ndshape.load[width=1](index).__int__()

    @always_inline("nodebug")
    fn store_int(inout self, index: Int, val: Int):
        """
        SIMD store dimensional information.
        """
        self.ndshape.store[width=1](index, val)


@register_passable("trivial")
struct NDArrayStride[dtype: DType = DType.int32](Stringable, Formattable):
    """Implements the NDArrayStride."""

    # Fields
    var ndoffset: Int
    var ndstride: UnsafePointer[Scalar[dtype]]
    var ndlen: Int

    @always_inline("nodebug")
    fn __init__(
        inout self, *stride: Int, offset: Int = 0
    ):  # maybe we should add checks for offset?
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(stride.__len__())
        for i in range(stride.__len__()):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: List[Int], offset: Int = 0):
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: VariadicList[Int], offset: Int = 0):
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: NDArrayStride[dtype]):
        self.ndoffset = stride.ndoffset
        self.ndlen = stride.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(stride.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride.ndstride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, stride: NDArrayStride[dtype], offset: Int = 0
    ):  # separated two methods to remove if condition
        self.ndoffset = offset
        self.ndlen = stride.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(stride.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride.ndstride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, *shape: Int, offset: Int = 0, order: String = "C"
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self, shape: List[Int], offset: Int = 0, order: String = "C"
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        owned shape: NDArrayShape,
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(shape.ndlen)
        memset_zero(self.ndstride, shape.ndlen)
        if order == "C":
            if shape.ndlen == 1:
                self.ndstride[0] = 1
            else:
                for i in range(shape.ndlen):
                    var temp: Int = 1
                    for j in range(i + 1, shape.ndlen):
                        temp = temp * shape[j]
                    self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    fn __copy__(inout self, other: Self):
        self.ndoffset = other.ndoffset
        self.ndlen = other.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(other.ndlen)
        memcpy(self.ndstride, other.ndstride, other.ndlen)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            return self.ndstride[index].__int__()
        else:
            return self.ndstride[self.ndlen + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            self.ndstride[index] = val
        else:
            self.ndstride[self.ndlen + index] = val

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self.ndlen

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        var result: String = "Stride: ["
        for i in range(self.ndlen):
            if i == self.ndlen - 1:
                result += self.ndstride[i].__str__()
            else:
                result += self.ndstride[i].__str__() + ", "
        result = result + "]"
        writer.write(result)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        for i in range(self.ndlen):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        for i in range(self.ndlen):
            if self[i] == val:
                return True
        return False

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        return self.ndstride.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        self.ndstride.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
        return self.ndstride.load[width=width](index).__int__()

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]):
        self.ndstride.store[width=width](index, val)
