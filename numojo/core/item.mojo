"""
Implements Item type.

`Item` is a series of `Int` on the heap.
"""

from builtin.int import index as convert_to_int
from memory import memcpy, memset_zero
from memory import UnsafePointer
from memory import memcmp
from os import abort
from sys import simd_width_of
from utils import Variant

from numojo.core.error import IndexError, ValueError
from numojo.core.traits.indexer_collection_element import (
    IndexerCollectionElement,
)


@register_passable
struct Item(
    ImplicitlyCopyable, Movable, Representable, Sized, Stringable, Writable
):
    """
    Represents a multi-dimensional index for array access.

    The `Item` struct is used to specify the coordinates of an element within an N-dimensional array.
    For example, `arr[Item(1, 2, 3)]` retrieves the element at position (1, 2, 3) in a 3D array.

    Each `Item` instance holds a sequence of integer indices, one for each dimension of the array.
    This allows for precise and flexible indexing into arrays of arbitrary dimensionality.

    Example:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        var arr = nm.arange[f32](0, 27).reshape(Shape(3, 3, 3))
        var value = arr[Item(1, 2, 3)]  # Accesses arr[1, 2, 3]
        ```

    Fields:
        _buf: Pointer to the buffer storing the indices.
        _ndim: Number of dimensions (length of the index tuple).
    """

    # Aliases
    comptime element_type: DType = DType.int
    """The data type of the Item elements."""
    comptime _origin: MutOrigin = MutOrigin.external
    """Internal origin of the Item instance."""

    # Fields
    var _buf: UnsafePointer[Scalar[Self.element_type], Self._origin]
    var ndim: Int

    # add constraint for ndim >= 0 for Item instance.
    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, *args: T):
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self._buf = alloc[Scalar[Self.element_type]](len(args))
        self.ndim = len(args)
        for i in range(len(args)):
            (self._buf + i).init_pointee_copy(convert_to_int(args[i]))

    @always_inline("nodebug")
    fn __init__[T: IndexerCollectionElement](out self, args: List[T]) raises:
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = alloc[Scalar[Self.element_type]](self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(convert_to_int(args[i]))

    @always_inline("nodebug")
    fn __init__(out self, args: VariadicList[Int]) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = alloc[Scalar[Self.element_type]](len(args))
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(args[i])

    @always_inline("nodebug")
    fn __init__(out self, ndim: Int):
        """Construct the Item with given length and initialize to zero.

        Args:
            ndim: The length of the tuple.
        """
        self.ndim = ndim
        self._buf = alloc[Scalar[Self.element_type]](ndim)
        memset_zero(self._buf, ndim)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.ndim = other.ndim
        self._buf = alloc[Scalar[Self.element_type]](self.ndim)
        memcpy(dest=self._buf, src=other._buf, count=self.ndim)

    @always_inline("nodebug")
    fn __del__(deinit self):
        if self.ndim > 0:
            self._buf.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.ndim

    fn normalize_index(self, index: Int) -> Int:
        """
        Normalizes the given index to be within the valid range.

        Args:
            index: The index to normalize.

        Returns:
            The normalized index.
        """
        var norm_idx: Int = index
        if norm_idx < 0:
            norm_idx += self.ndim
        return norm_idx

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
        var index: Int = convert_to_int(idx)
        if index >= self.ndim or index < -self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{} , {}).").format(
                        convert_to_int(idx), -self.ndim, self.ndim
                    ),
                    suggestion=String(
                        "Use indices in [-ndim, ndim) (negative indices wrap)."
                    ),
                    location=String("Item.__getitem__"),
                )
            )
        var normalized_idx: Int = self.normalize_index(convert_to_int(idx))
        return Int(self._buf[normalized_idx])

    @always_inline("nodebug")
    fn __getitem__(self, slice_index: Slice) raises -> Self:
        """
        Return a sliced view of the item as a new Item.
        Delegates normalization & validation to _compute_slice_params.

        Args:
            slice_index: The slice to extract.

        Returns:
            A new Item containing the sliced values.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(10, 20, 30, 40, 50)
        print(item[1:4])  # Item: (20, 30, 40)
        print(item[::2])  # Item: (10, 30, 50)
        ```
        """
        var updated_slice: Tuple[Int, Int, Int] = self._compute_slice_params(
            slice_index
        )
        var start = updated_slice[0]
        var step = updated_slice[1]
        var length = updated_slice[2]

        if length <= 0:
            raise Error(
                ShapeError(
                    message="Provided slice results in an empty Item.",
                    suggestion=(
                        "Adjust slice parameters to obtain non-empty result."
                    ),
                    location="Item.__getitem__(self, slice_list: Slice)",
                )
            )

        var result: Item = Self(ndim=length)
        var idx: Int = start
        for i in range(length):
            (result._buf + i).init_pointee_copy(self._buf[idx])
            idx += step
        return result^

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
        var norm_idx: Int = self.normalize_index(convert_to_int(idx))
        if norm_idx < 0 or norm_idx >= self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{} , {}).").format(
                        convert_to_int(idx), -self.ndim, self.ndim
                    ),
                    suggestion=String(
                        "Use indices in [-ndim, ndim) (negative indices wrap)."
                    ),
                    location=String("Item.__setitem__"),
                )
            )

        self._buf[norm_idx] = index(val)

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
        var result: String = "Item" + String(self)
        return result

    fn __str__(self) -> String:
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf[i])
            if i < self.ndim - 1:
                result += ", "
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Coordinates: " + String(self) + "  ")

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two items have identical dimensions and values.

        Args:
            other: The item to compare with.

        Returns:
            True if both items have identical dimensions and values.
        """
        if self.ndim != other.ndim:
            return False
        if memcmp(self._buf, other._buf, self.ndim) != 0:
            return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two items have different dimensions or values.

        Args:
            other: The item to compare with.

        Returns:
            True if both items do not have identical dimensions or values.
        """
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) -> Bool:
        """
        Checks if the given value is present in the item.

        Args:
            val: The value to search for.

        Returns:
            True if the given value is present in the item.
        """
        for i in range(self.ndim):
            if self._buf[i] == val:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#
    @always_inline("nodebug")
    fn deep_copy(read self) raises -> Self:
        """
        Returns a deep copy of the item.

        Returns:
            A new Item with the same values.
        """
        var res: Item = Item(ndim=self.ndim)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)
        return res^

    fn swapaxes(self, axis1: Int, axis2: Int) raises -> Self:
        """
        Returns a new item with the given axes swapped.

        Args:
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            A new item with the given axes swapped.
        """
        var res: Item = Item(ndim=self.ndim)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)
        res[axis1] = self[axis2]
        res[axis2] = self[axis1]
        return res

    fn join(self, *others: Self) raises -> Self:
        """
        Join multiple items into a single item.

        Args:
            others: Variable number of Item objects.

        Returns:
            A new Item object with all values concatenated.

        Examples:
        ```mojo
        from numojo.prelude import *
        var item1 = Item(1, 2)
        var item2 = Item(3, 4)
        var item3 = Item(5)
        var joined = item1.join(item2, item3)
        print(joined)  # Item at index: (1,2,3,4,5)
        ```
        """
        var total_dims: Int = self.ndim
        for i in range(len(others)):
            total_dims += others[i].ndim

        var new_item: Item = Item(ndim=total_dims)

        var index: UInt = 0
        for i in range(self.ndim):
            (new_item._buf + index).init_pointee_copy(self[i])
            index += 1

        for i in range(len(others)):
            for j in range(others[i].ndim):
                (new_item._buf + index).init_pointee_copy(others[i][j])
                index += 1

        return new_item^

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _flip(self) raises -> Self:
        """
        Returns a new item by flipping the items.
        ***UNSAFE!*** No boundary check!

        Returns:
            A new item with the items flipped.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(1, 2, 3)
        print(item)          # Item: (1, 2, 3)
        print(item._flip())  # Item: (3, 2, 1)
        ```
        """
        var res: Item = Item(ndim=self.ndim)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)
        for i in range(res.ndim):
            res._buf[i] = self._buf[self.ndim - 1 - i]
        return res^

    fn _move_axis_to_end(self, var axis: Int) raises -> Self:
        """
        Returns a new item by moving the value of axis to the end.
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to move. It should be in `[-ndim, ndim)`.

        Returns:
            A new item with the specified axis moved to the end.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(10, 20, 30)
        print(item._move_axis_to_end(0))  # Item: (20, 30, 10)
        print(item._move_axis_to_end(1))  # Item: (10, 30, 20)
        ```
        """
        if axis < 0:
            axis += self.ndim

        var res: Item = Item(ndim=self.ndim)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)

        if axis == self.ndim - 1:
            return res^

        var value: Scalar[Self.element_type] = res._buf[axis]
        for i in range(axis, res.ndim - 1):
            res._buf[i] = res._buf[i + 1]
        res._buf[res.ndim - 1] = value
        return res^

    fn _pop(self, axis: Int) raises -> Self:
        """
        Drops information of certain axis.
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to drop. It should be in `[0, ndim)`.

        Returns:
            A new item with the item at the given axis (index) dropped.
        """
        var res: Item = Item(ndim=self.ndim - 1)
        memcpy(dest=res._buf, src=self._buf, count=axis)
        memcpy(
            dest=res._buf + axis,
            src=self._buf.offset(axis + 1),
            count=self.ndim - axis - 1,
        )
        return res^

    fn _extend(self, *values: Int) raises -> Self:
        """
        Extend the item by additional values.
        ***UNSAFE!*** No boundary check!

        Args:
            values: Additional values to append.

        Returns:
            A new Item object with the extended values.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(1, 2, 3)
        var extended = item._extend(4, 5)
        print(extended)  # Item: (1, 2, 3, 4, 5)
        ```
        """
        var total_dims: Int = self.ndim + len(values)
        var new_item: Item = Item(ndim=total_dims)

        var offset: UInt = 0
        for i in range(self.ndim):
            (new_item._buf + offset).init_pointee_copy(self[i])
            offset += 1
        for value in values:
            (new_item._buf + offset).init_pointee_copy(value)
            offset += 1

        return new_item^

    fn _compute_slice_params(
        self, slice_index: Slice
    ) raises -> Tuple[Int, Int, Int]:
        """
        Compute normalized slice parameters (start, step, length).

        Args:
            slice_index: The slice to compute parameters for.

        Returns:
            A tuple of (start, step, length).

        Raises:
            Error: If the slice step is zero.
        """
        var n = self.ndim
        if n == 0:
            return (0, 1, 0)

        var step = slice_index.step.or_else(1)
        if step == 0:
            raise Error(
                ValueError(
                    message="Slice step cannot be zero.",
                    suggestion="Use a non-zero step value.",
                    location="Item._compute_slice_params",
                )
            )

        var start: Int
        var stop: Int
        if step > 0:
            start = slice_index.start.or_else(0)
            stop = slice_index.end.or_else(n)
        else:
            start = slice_index.start.or_else(n - 1)
            stop = slice_index.end.or_else(-1)

        if start < 0:
            start += n
        if stop < 0:
            stop += n

        if step > 0:
            if start < 0:
                start = 0
            if start > n:
                start = n
            if stop < 0:
                stop = 0
            if stop > n:
                stop = n
        else:
            if start >= n:
                start = n - 1
            if start < -1:
                start = -1
            if stop >= n:
                stop = n - 1
            if stop < -1:
                stop = -1

        var length: Int = 0
        if step > 0:
            if start < stop:
                length = Int((stop - start + step - 1) / step)
        else:
            if start > stop:
                var neg_step = -step
                length = Int((start - stop + neg_step - 1) / neg_step)

        return (start, step, length)

    fn load[
        width: Int = 1
    ](self, idx: Int) raises -> SIMD[Self.element_type, width]:
        """
        Load a SIMD vector from the Item at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.

        Raises:
            Error: If the load exceeds the bounds of the Item.
        """
        if idx < 0 or idx + width > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Load operation out of bounds: idx={} width={} ndim={}"
                    ).format(idx, width, self.ndim),
                    suggestion=(
                        "Ensure that idx and width are within valid range."
                    ),
                    location="Item.load",
                )
            )

        return self._buf.load[width=width](idx)

    fn store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]) raises:
        """
        Store a SIMD vector into the Item at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.

        Raises:
            Error: If the store exceeds the bounds of the Item.
        """
        if idx < 0 or idx + width > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Store operation out of bounds: idx={} width={} ndim={}"
                    ).format(idx, width, self.ndim),
                    suggestion=(
                        "Ensure that idx and width are within valid range."
                    ),
                    location="Item.store",
                )
            )

        self._buf.store[width=width](idx, value)

    fn unsafe_load[
        width: Int = 1
    ](self, idx: Int) -> SIMD[Self.element_type, width]:
        """
        Unsafely load a SIMD vector from the Item at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.
        """
        return self._buf.load[width=width](idx)

    fn unsafe_store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]):
        """
        Unsafely store a SIMD vector into the Item at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.
        """
        self._buf.store[width=width](idx, value)


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
        self.index = 0 if forward else length - 1
        self.length = length
        self.item = item

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index >= 0

    fn __next__(mut self) raises -> Scalar[DType.int]:
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
