"""
Implements Item type.

`Item` is a series of `Int` on the heap.
"""

from builtin.type_aliases import Origin
from memory import UnsafePointer, memset_zero, memcpy
from os import abort
from sys import simd_width_of
from utils import Variant

from numojo.core.traits.indexer_collection_element import (
    IndexerCollectionElement,
)

# simple alias for users. Use `Item` internally.
alias item = Item


@register_passable
struct Item(ImplicitlyCopyable, Movable, Stringable, Writable):
    """
    Specifies the indices of an item of an array.
    """

    var _buf: UnsafePointer[Int]
    var ndim: Int

    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, *args: T):
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self._buf = UnsafePointer[Int]().alloc(args.__len__())
        self.ndim = args.__len__()
        for i in range(args.__len__()):
            self._buf[i] = index(args[i])

    @always_inline("nodebug")
    fn __init__[T: IndexerCollectionElement](out self, args: List[T]) raises:
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(index(args[i]))

    @always_inline("nodebug")
    fn __init__(out self, args: VariadicList[Int]) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(Int(args[i]))

    fn __init__(
        out self,
        *,
        ndim: Int,
        initialized: Bool,
    ) raises:
        """
        Construct Item with number of dimensions.
        This method is useful when you want to create a Item with given ndim
        without knowing the Item values.

        Args:
            ndim: Number of dimensions.
            initialized: Whether the shape is initialized.
                If yes, the values will be set to 0.
                If no, the values will be uninitialized.

        Raises:
            Error: If the number of dimensions is negative.
        """
        if ndim < 0:
            raise Error(
                IndexError(
                    message=String(
                        "Invalid ndim: got {}; must be >= 0."
                    ).format(ndim),
                    suggestion=String(
                        "Pass a non-negative dimension count when constructing"
                        " Item."
                    ),
                    location=String("Item.__init__(ndim: Int)"),
                )
            )

        self.ndim = ndim
        self._buf = UnsafePointer[Int]().alloc(ndim)
        if initialized:
            for i in range(ndim):
                (self._buf + i).init_pointee_copy(0)

    fn __init__(out self, idx: Int, shape: NDArrayShape) raises:
        """
        Get indices of the i-th item of the array of the given shape.
        The item traverse the array in C-order.

        Args:
            idx: The i-th item of the array.
            shape: The strides of the array.

        Examples:

        The following example demonstrates how to get the indices (coordinates)
        of the 123-th item of a 3D array with shape (20, 30, 40).

        ```console
        >>> from numojo.prelude import *
        >>> var item = Item(123, Shape(20, 30, 40))
        >>> print(item)
        Item at index: (0,3,3)  Length: 3
        ```
        """

        if (idx < 0) or (idx >= shape.size_of_array()):
            raise Error(
                IndexError(
                    message=String(
                        "Linear index {} out of range [0, {})."
                    ).format(idx, shape.size_of_array()),
                    suggestion=String(
                        "Ensure 0 <= idx < total size ({})."
                    ).format(shape.size_of_array()),
                    location=String(
                        "Item.__init__(idx: Int, shape: NDArrayShape)"
                    ),
                )
            )

        self.ndim = shape.ndim
        self._buf = UnsafePointer[Int]().alloc(self.ndim)

        var strides = NDArrayStrides(shape, order="C")
        var remainder = idx

        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(remainder // strides._buf[i])
            remainder %= strides._buf[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.ndim = other.ndim
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        memcpy(self._buf, other._buf, self.ndim)

    @always_inline("nodebug")
    fn __del__(deinit self):
        self._buf.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __getitem__[T: Indexer](self, idx: T) raises -> Int:
        """Gets the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            idx: The index of the value to get.

        Returns:
            The value at the specified index.
        """

        var normalized_idx: Int = index(idx)
        if normalized_idx < 0:
            normalized_idx = index(idx) + self.ndim

        if normalized_idx < 0 or normalized_idx >= self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{} , {}).").format(
                        index(idx), -self.ndim, self.ndim
                    ),
                    suggestion=String(
                        "Use indices in [-ndim, ndim) (negative indices wrap)."
                    ),
                    location=String("Item.__getitem__"),
                )
            )

        return self._buf[normalized_idx]

    @always_inline("nodebug")
    fn __setitem__[T: Indexer, U: Indexer](self, idx: T, val: U) raises:
        """Set the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.
            U: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            idx: The index of the value to set.
            val: The value to set.
        """

        var normalized_idx: Int = index(idx)
        if normalized_idx < 0:
            normalized_idx = index(idx) + self.ndim

        if normalized_idx < 0 or normalized_idx >= self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{} , {}).").format(
                        index(idx), -self.ndim, self.ndim
                    ),
                    suggestion=String(
                        "Use indices in [-ndim, ndim) (negative indices wrap)."
                    ),
                    location=String("Item.__setitem__"),
                )
            )

        self._buf[normalized_idx] = index(val)

    fn __iter__(self) raises -> _ItemIter:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _ItemIter(
            item=self,
            length=self.ndim,
        )

    fn __repr__(self) -> String:
        var result: String = "numojo.Item" + String(self)
        return result

    fn __str__(self) -> String:
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf[i])
            if i < self.ndim - 1:
                result += ","
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Item at index: "
            + String(self)
            + "  "
            + "Length: "
            + String(self.ndim)
        )

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    fn offset(self, strides: NDArrayStrides) -> Int:
        """
        Calculates the offset of the item according to strides.

        Args:
            strides: The strides of the array.

        Returns:
            The offset of the item.

        Examples:

        ```mojo
        from numojo.prelude import *
        var item = Item(1, 2, 3)
        var strides = nm.Strides(4, 3, 2)
        print(item.offset(strides))
        # This prints `16`.
        ```
        .
        """

        var offset: Int = 0
        for i in range(self.ndim):
            offset += self._buf[i] * strides._buf[i]
        return offset


struct _ItemIter[
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for Item.

    Parameters:
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var item: Item
    var length: Int

    fn __init__(
        out self,
        item: Item,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.item = item

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __next__(mut self) raises -> Scalar[DType.index]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.item.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.item.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index
