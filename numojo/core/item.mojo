"""
Implements Item type.

`Item` is a series of `Int` on the heap.
"""

from sys import simdwidthof
from utils import Variant
from builtin.type_aliases import Origin
from memory import UnsafePointer, memset_zero, memcpy

alias item = Item


struct Item(CollectionElement):
    alias dtype: DType = DType.index
    alias width = simdwidthof[Self.dtype]()
    var _buf: UnsafePointer[Scalar[Self.dtype]]
    var len: Int

    @always_inline("nodebug")
    fn __init__(mut self, owned *args: Scalar[Self.dtype]):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self._buf = UnsafePointer[Scalar[Self.dtype]]().alloc(args.__len__())
        self.len = args.__len__()
        for i in range(args.__len__()):
            self._buf[i] = args[i]

    @always_inline("nodebug")
    fn __init__(mut self, owned *args: Int):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self._buf = UnsafePointer[Scalar[Self.dtype]]().alloc(args.__len__())
        self.len = args.__len__()
        for i in range(args.__len__()):
            self._buf[i] = args[i]

    @always_inline("nodebug")
    fn __init__(
        mut self, owned args: Variant[List[Int], VariadicList[Int]]
    ) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        if args.isa[List[Int]]():
            self.len = args[List[Int]].__len__()
            self._buf = UnsafePointer[Scalar[Self.dtype]]().alloc(self.len)
            for i in range(self.len):
                self._buf[i] = args[List[Int]][i]
        elif args.isa[VariadicList[Int]]():
            self.len = args[VariadicList[Int]].__len__()
            self._buf = UnsafePointer[Scalar[Self.dtype]]().alloc(self.len)
            for i in range(self.len):
                self._buf[i] = args[VariadicList[Int]][i]
        else:
            raise Error("Invalid type")

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self._buf = UnsafePointer[Scalar[Self.dtype]]().alloc(other.__len__())
        self.len = other.len
        for i in range(other.__len__()):
            self._buf[i] = other[i]

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned other: Self):
        """Move construct the tuple.

        Args:
            other: The tuple to move.
        """
        self._buf = other._buf
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
        return int(self._buf[index])

    @always_inline("nodebug")
    fn __setitem__(self, index: Int, val: Int):
        """Set the value at the specified index.

        Args:
            index: The index of the value to set.
            val: The value to set.
        """
        self._buf[index] = val

    fn __iter__(self) raises -> _ItemIter[__origin_of(self)]:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _ItemIter[__origin_of(self)](
            array=self,
            length=self.len,
        )

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Item: " + self.str() + "\n" + "Length: " + str(self.len))

    fn str(self) -> String:
        var result: String = "["
        for i in range(self.len):
            result += str(self._buf[i])
            if i < self.len - 1:
                result += ", "
        result += "]"
        return result

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[Self.dtype, width]:
        if index + width - 1 > self.len:
            raise Error("Index out of bounds")
        return self._buf.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](mut self, index: Int, val: SIMD[Self.dtype, width]) raises:
        if index + width - 1 > self.len:
            raise Error("Index out of bounds")
        self._buf.store(index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> SIMD[Self.dtype, width]:
        return self._buf.load[width=width](index)

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](mut self, index: Int, val: SIMD[Self.dtype, width]):
        self._buf.store(index, val)


@value
struct _ItemIter[
    is_mutable: Bool, //,
    lifetime: Origin[is_mutable],
    forward: Bool = True,
]:
    """Iterator for Item.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying Item data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Item
    var length: Int

    fn __init__(
        mut self,
        array: Item,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> Scalar[DType.index]:
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
