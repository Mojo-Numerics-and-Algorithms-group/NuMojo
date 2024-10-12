from utils import Variant
from builtin.type_aliases import AnyLifetime
from memory import memset_zero, memcpy


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
