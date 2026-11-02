# ===----------------------------------------------------------------------=== #
# NuMojo: NDLayout type
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
NDLayout (numojo.core.layout.ndlayout).
=======================================
Compact owned layout metadata for N-dimensional arrays.

Exports
-------
- `NDLayout`: Shape, strides, offset, and size stored as one value.
- `LayoutIndex`: Integer, slice, ellipsis, or new-axis layout index.
- `Shape`: Small-buffer-optimized shape metadata.
- `Strides`: Small-buffer-optimized stride metadata.
- `INLINE_LAYOUT_RANK`: Rank stored without heap allocation.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.builtin.type_aliases import EllipsisType
from std.collections import Array
from std.utils import Variant

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.layout.array_methods import NewAxis


comptime INLINE_LAYOUT_RANK = 8
"""Number of shape or stride entries stored inline in an `NDLayout`."""


struct _SmallIntArray[name: StaticString](Copyable, Movable, Sized):
    """Owned integer metadata with an eight-element inline buffer."""

    var _inline: Array[Int, INLINE_LAYOUT_RANK]
    var _spill: List[Int]
    var _length: Int

    def __init__(out self, *values: Int):
        """Construct metadata from zero or more integer values.

        Args:
            values: Values to store.

        Examples:
            ```mojo
            var values = _SmallIntArray(1, 2, 3)
            ```
        """
        self._inline = Array[Int, INLINE_LAYOUT_RANK](fill=0)
        self._spill = List[Int]()
        self._length = len(values)
        if len(values) > INLINE_LAYOUT_RANK:
            self._spill = List[Int](length=len(values), fill=0)
        for i in range(len(values)):
            self.unsafe_set(i, values[i])

    @staticmethod
    def _with_length(length: Int) raises -> Self:
        """Construct zero-initialized metadata with the requested length.

        Args:
            length: Number of entries.

        Returns:
            Zero-initialized metadata.

        Raises:
            NumojoError: If `length` is negative.
        """
        if length < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="Metadata length must be non-negative.",
                    location="_SmallIntArray._with_length(length)",
                )
            )
        var result = Self()
        result._length = length
        if length > INLINE_LAYOUT_RANK:
            result._spill = List[Int](length=length, fill=0)
        return result^

    def __init__(out self, values: List[Int]) raises:
        """Construct metadata by copying a list of values.

        Args:
            values: Values to copy.

        Examples:
            ```mojo
            var values = _SmallIntArray([1, 2, 3])
            ```
        """
        self = Self._with_length(len(values))
        for i in range(len(values)):
            self.unsafe_set(i, values[i])

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self._length

    @always_inline("nodebug")
    def is_inline(self) -> Bool:
        """Return whether the values are stored inline.

        Returns:
            True when the length does not exceed `INLINE_LAYOUT_RANK`.

        Examples:
            ```mojo
            var inline = _SmallIntArray._with_length(3).is_inline()
            ```
        """
        return self._length <= INLINE_LAYOUT_RANK

    @always_inline("nodebug")
    def __getitem__(self, index: Int) raises -> Int:
        if index < 0 or index >= self._length:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} is out of bounds for length {}."
                    ).format(index, self._length),
                    location="_SmallIntArray.__getitem__(index)",
                )
            )
        if self.is_inline():
            return self._inline[index]
        return self._spill[index]

    @always_inline("nodebug")
    def __setitem__(mut self, index: Int, value: Int) raises:
        if index < 0 or index >= self._length:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} is out of bounds for length {}."
                    ).format(index, self._length),
                    location="_SmallIntArray.__setitem__(index, value)",
                )
            )
        if self.is_inline():
            self._inline[index] = value
        else:
            self._spill[index] = value

    @always_inline("nodebug")
    def unsafe_get(self, index: Int) -> Int:
        """Return the value at `index` without bounds checking.

        Args:
            index: Index of the value to return.

        Returns:
            The value at `index`.

        Examples:
            ```mojo
            var value = _SmallIntArray([1, 2, 3]).unsafe_get(1)
            ```
        """
        if self.is_inline():
            return self._inline[index]
        return self._spill[index]

    @always_inline("nodebug")
    def unsafe_set(mut self, index: Int, value: Int):
        """Set the value at `index` without bounds checking.

        Args:
            index: Index of the value to set.
            value: Value to store.

        Examples:
            ```mojo
            var values = _SmallIntArray([1, 2, 3])
            values.unsafe_set(1, 42)
            ```
        """
        if self.is_inline():
            self._inline[index] = value
        else:
            self._spill[index] = value

    def tolist(self) -> List[Int]:
        """Copy the metadata into a list.

        Returns:
            A list containing the metadata values.

        Examples:
            ```mojo
            var values = _SmallIntArray([1, 2, 3]).tolist()
            ```
        """
        var result = List[Int](capacity=self._length)
        for i in range(self._length):
            result.append(self.unsafe_get(i))
        return result^

    def __str__(self) -> String:
        """Return a string representation of the metadata."""
        var result = String("[")
        for i in range(self._length):
            if i > 0:
                result += ", "
            result += String(self.unsafe_get(i))
        return result + "]"


comptime Shape = _SmallIntArray["Shape"]
"""Owned shape metadata with an eight-element inline buffer."""

comptime Strides = _SmallIntArray["Strides"]
"""Owned stride metadata with an eight-element inline buffer."""


struct LayoutIndex(ImplicitlyCopyable):
    """Index value used by `NDLayout.sliced`.

    A layout index can select an integer position, slice an axis, expand an
    ellipsis, or insert a singleton axis.
    """

    var value: Variant[Int, Slice, EllipsisType, NewAxis]

    @implicit
    def __init__(out self, index: Int):
        """Construct an integer layout index.

        Args:
            index: Position to select.

        Examples:
            ```mojo
            var index = LayoutIndex(1)
            ```
        """
        self.value = Variant[Int, Slice, EllipsisType, NewAxis](index)

    @implicit
    def __init__(out self, slice_value: Slice):
        """Construct a slice layout index.

        Args:
            slice_value: Slice to apply.

        Examples:
            ```mojo
            var index = LayoutIndex(Slice(0, 4, 2))
            ```
        """
        self.value = Variant[Int, Slice, EllipsisType, NewAxis](slice_value)

    @implicit
    def __init__(out self, ellipsis: EllipsisType):
        """Construct an ellipsis layout index.

        Args:
            ellipsis: Ellipsis value.

        Examples:
            ```mojo
            var index = LayoutIndex(...)
            ```
        """
        self.value = Variant[Int, Slice, EllipsisType, NewAxis](ellipsis)

    @implicit
    def __init__(out self, axis: NewAxis):
        """Construct a singleton-axis insertion index.

        Args:
            axis: New-axis marker.

        Examples:
            ```mojo
            from numojo.core.layout import newaxis
            var index = LayoutIndex(newaxis)
            ```
        """
        self.value = Variant[Int, Slice, EllipsisType, NewAxis](axis)

    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        step: Optional[Int] = None,
        *,
        __slice_literal__: NoneType,
    ):
        """Construct an index from slice-literal components.

        Args:
            start: Optional inclusive start.
            end: Optional exclusive end.
            step: Optional step.
            __slice_literal__: Compiler-provided slice-literal marker.

        Examples:
            ```mojo
            var index: LayoutIndex = 1:5:2
            ```
        """
        self.value = Variant[Int, Slice, EllipsisType, NewAxis](
            Slice(start, end, step)
        )


struct NDLayout(Copyable, Movable, Sized, Writable):
    """Owned shape, stride, offset, and size metadata.

    Shape and stride entries are stored inline through rank eight and spill to
    heap-backed lists only for higher-rank arrays. Keeping the complete layout
    in one value makes owner-to-view metadata copies allocation-free for common
    array ranks.
    """

    var _shape: Shape
    var _strides: Strides
    var offset: Int
    var size: Int

    def __init__(out self):
        """Construct an empty, unallocated layout.

        Examples:
            ```mojo
            var layout = NDLayout()
            ```
        """
        self._shape = Shape()
        self._strides = Strides()
        self.offset = 0
        self.size = 0

    def __init__(out self, shape: Shape, order: String = "C") raises:
        """Construct a contiguous layout from a shape and memory order.

        Args:
            shape: Array dimensions.
            order: Memory order, either `"C"` or `"F"`.

        Raises:
            NumojoError: If `order` is not `"C"` or `"F"`.

        Examples:
            ```mojo
            var layout = NDLayout(Shape([2, 3]))
            ```
        """
        self._shape = shape.copy()
        self._strides = Strides._with_length(len(shape))
        self.offset = 0
        self.size = 1

        var stride = 1
        if order == "C":
            for i in range(len(shape) - 1, -1, -1):
                var extent = shape.unsafe_get(i)
                Self._validate_extent(extent, i)
                self._strides[i] = stride
                Self._check_product(self.size, extent)
                self.size *= extent
                stride = self.size
        elif order == "F":
            for i in range(len(shape)):
                var extent = shape.unsafe_get(i)
                Self._validate_extent(extent, i)
                self._strides[i] = stride
                Self._check_product(self.size, extent)
                self.size *= extent
                stride = self.size
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Invalid order '{}'; expected 'C' or 'F'. Use 'C' for"
                        " row-major or 'F' for column-major layout."
                    ).format(order),
                    location="NDLayout.__init__(shape, order)",
                )
            )

    def __init__(out self, shape: List[Int], order: String = "C") raises:
        """Construct a contiguous layout from shape dimensions.

        Args:
            shape: Array dimensions.
            order: Memory order, either `"C"` or `"F"`.

        Raises:
            NumojoError: If a dimension or `order` is invalid.

        Examples:
            ```mojo
            var layout = NDLayout([2, 3])
            ```
        """
        self = Self(Shape(shape), order=order)

    def __init__(
        out self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
    ) raises:
        """Construct a layout with explicit strides and offset.

        Args:
            shape: Array dimensions.
            strides: Element strides for each dimension.
            offset: First logical element's offset in the storage buffer.

        Raises:
            NumojoError: If the shape and stride ranks differ.

        Examples:
            ```mojo
            var layout = NDLayout(
                Shape([2, 3]), Strides([3, 1]), offset=0
            )
            ```
        """
        if len(shape) != len(strides):
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Shape rank {} does not match stride rank {}."
                    ).format(len(shape), len(strides)),
                    location="NDLayout.__init__(shape, strides, offset)",
                )
            )

        self._shape = shape.copy()
        self._strides = strides.copy()
        self.offset = offset
        self.size = 1
        for i in range(len(shape)):
            var extent = shape.unsafe_get(i)
            Self._validate_extent(extent, i)
            Self._check_product(self.size, extent)
            self.size *= extent

    @staticmethod
    @always_inline("nodebug")
    def _validate_extent(extent: Int, axis: Int) raises:
        if extent < 0:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Shape dimension at index {} must be non-negative, got"
                        " {}."
                    ).format(axis, extent),
                    location="NDLayout.__init__",
                )
            )

    @staticmethod
    @always_inline("nodebug")
    def _check_product(current: Int, extent: Int) raises:
        if extent != 0 and current > Int.MAX // extent:
            raise Error(
                NumojoError(
                    category="shape",
                    message="Shape element count exceeds the Int range.",
                    location="NDLayout.__init__",
                )
            )

    @staticmethod
    def _shape_product(shape: Shape) raises -> Int:
        var result = 1
        for axis in range(len(shape)):
            var extent = shape.unsafe_get(axis)
            Self._validate_extent(extent, axis)
            Self._check_product(result, extent)
            result *= extent
        return result

    @staticmethod
    def _list_product(shape: List[Int]) raises -> Int:
        var result = 1
        for axis in range(len(shape)):
            var extent = shape[axis]
            Self._validate_extent(extent, axis)
            Self._check_product(result, extent)
            result *= extent
        return result

    def __init__(
        out self,
        shape: List[Int],
        strides: List[Int],
        offset: Int = 0,
    ) raises:
        """Construct a layout from explicit shape and stride lists.

        Args:
            shape: Array dimensions.
            strides: Element strides for each dimension.
            offset: First logical element's offset in the storage buffer.

        Raises:
            NumojoError: If the metadata is invalid or ranks differ.

        Examples:
            ```mojo
            var layout = NDLayout([2, 3], [3, 1])
            ```
        """
        self = Self(Shape(shape), Strides(strides), offset)

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Return the number of dimensions."""
        return len(self._shape)

    @always_inline("nodebug")
    def is_inline(self) -> Bool:
        """Return whether shape and strides use their inline storage.

        Returns:
            True when the rank does not exceed `INLINE_LAYOUT_RANK`.

        Examples:
            ```mojo
            var inline = NDLayout([2, 3]).is_inline()
            ```
        """
        return self._shape.is_inline()

    def has_same_shape(self, other: Self) -> Bool:
        """Return whether two layouts have equal shape dimensions.

        Args:
            other: Layout to compare.

        Returns:
            True when both layouts have the same rank and extents.

        Examples:
            ```mojo
            var same = NDLayout([2, 3]).has_same_shape(NDLayout([2, 3]))
            ```
        """
        if len(self) != len(other):
            return False
        for axis in range(len(self)):
            if self.unsafe_shape_at(axis) != other.unsafe_shape_at(axis):
                return False
        return True

    @staticmethod
    @always_inline("nodebug")
    def _normalize_index(var index: Int, extent: Int) raises -> Int:
        if index < 0:
            index += extent
        if index < 0 or index >= extent:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} is out of bounds for extent {}."
                    ).format(index, extent),
                    location="NDLayout._normalize_index(index, extent)",
                )
            )
        return index

    def position(self, indices: List[Int]) raises -> Int:
        """Return the physical storage offset for N-D coordinates.

        Args:
            indices: One coordinate per layout axis. Negative indices wrap.

        Returns:
            Element offset from the start of the underlying storage.

        Raises:
            NumojoError: If the coordinate rank or an index is invalid.

        Examples:
            ```mojo
            var offset = NDLayout([2, 3]).position([1, 2])
            ```
        """
        if len(indices) != len(self):
            raise Error(
                NumojoError(
                    category="index",
                    message=String("Expected {} indices but got {}.").format(
                        len(self), len(indices)
                    ),
                    location="NDLayout.position(indices)",
                )
            )
        var result = self.offset
        for axis in range(len(indices)):
            var index = Self._normalize_index(
                indices[axis], self.unsafe_shape_at(axis)
            )
            result += index * self.unsafe_stride_at(axis)
        return result

    def flat_position(self, var index: Int) raises -> Int:
        """Return the physical storage offset for a C-order flat index.

        Args:
            index: Flat logical index. Negative indices wrap.

        Returns:
            Element offset from the start of the underlying storage.

        Raises:
            NumojoError: If `index` is out of bounds.

        Examples:
            ```mojo
            var offset = NDLayout([2, 3]).flat_position(4)
            ```
        """
        index = Self._normalize_index(index, self.size)
        var result = self.offset
        for reverse_axis in range(len(self)):
            var axis = len(self) - 1 - reverse_axis
            var extent = self.unsafe_shape_at(axis)
            result += (index % extent) * self.unsafe_stride_at(axis)
            index //= extent
        return result

    def slice(
        self,
        var axis: Int,
        start: Int,
        count: Int,
        step: Int = 1,
    ) raises -> Self:
        """Return a layout sliced along one axis.

        Args:
            axis: Axis to slice. Negative axes wrap.
            start: Index of the first selected element.
            count: Number of selected elements.
            step: Distance between selected elements.

        Returns:
            A layout describing the sliced view.

        Raises:
            NumojoError: If an argument is invalid or the slice exceeds the axis.

        Examples:
            ```mojo
            var sliced = NDLayout([4, 5]).slice(1, 1, 2, 2)
            ```
        """
        axis = Self._normalize_index(axis, len(self))
        if count < 0 or step == 0 or step == Int.MIN:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Slice count must be non-negative and step nonzero."
                    ),
                    location="NDLayout.slice(axis, start, count, step)",
                )
            )

        var result = self.copy()
        if count > 0:
            var first = Self._normalize_index(start, self.unsafe_shape_at(axis))
            if step > 0:
                if count - 1 > (self.unsafe_shape_at(axis) - 1 - first) // step:
                    raise Error(
                        NumojoError(
                            category="index",
                            message="Slice exceeds the selected axis.",
                            location="NDLayout.slice(axis, start, count, step)",
                        )
                    )
            elif count - 1 > first // (-step):
                raise Error(
                    NumojoError(
                        category="index",
                        message="Slice exceeds the selected axis.",
                        location="NDLayout.slice(axis, start, count, step)",
                    )
                )
            if self.size > 0:
                result.offset += first * self.unsafe_stride_at(axis)

        result._strides.unsafe_set(
            axis, self.unsafe_stride_at(axis) * step if count > 1 else 0
        )
        result._shape.unsafe_set(axis, count)
        result.size = Self._shape_product(result._shape)
        return result^

    def select(self, var axis: Int, index: Int) raises -> Self:
        """Return a layout with one indexed axis removed.

        Args:
            axis: Axis to remove. Negative axes wrap.
            index: Element selected from that axis. Negative indices wrap.

        Returns:
            A rank-one-lower layout at the selected element.

        Raises:
            NumojoError: If the axis or index is out of bounds.

        Examples:
            ```mojo
            var row = NDLayout([2, 3]).select(0, 1)
            ```
        """
        axis = Self._normalize_index(axis, len(self))
        var normalized = Self._normalize_index(
            index, self.unsafe_shape_at(axis)
        )
        var new_offset = self.offset
        if self.size > 0:
            new_offset += normalized * self.unsafe_stride_at(axis)

        var shape = List[Int](capacity=len(self) - 1)
        var strides = List[Int](capacity=len(self) - 1)
        for current_axis in range(len(self)):
            if current_axis != axis:
                shape.append(self.unsafe_shape_at(current_axis))
                strides.append(self.unsafe_stride_at(current_axis))
        return Self(shape, strides, new_offset)

    def transpose(self, axes: List[Int]) raises -> Self:
        """Return a layout with permuted axes.

        Args:
            axes: Complete axis permutation. Negative axes wrap.

        Returns:
            A layout with shape and strides permuted together.

        Raises:
            NumojoError: If `axes` is not a valid complete permutation.

        Examples:
            ```mojo
            var transposed = NDLayout([2, 3, 4]).transpose([2, 0, 1])
            ```
        """
        if len(axes) != len(self):
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Transpose requires {} axes but got {}."
                    ).format(len(self), len(axes)),
                    location="NDLayout.transpose(axes)",
                )
            )

        var shape = Shape._with_length(len(self))
        var strides = Strides._with_length(len(self))
        var seen = List[Bool](length=len(self), fill=False)
        for result_axis in range(len(self)):
            var source_axis = Self._normalize_index(
                axes[result_axis], len(self)
            )
            if seen[source_axis]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Axis {} appears more than once."
                        ).format(axes[result_axis]),
                        location="NDLayout.transpose(axes)",
                    )
                )
            seen[source_axis] = True
            shape.unsafe_set(result_axis, self.unsafe_shape_at(source_axis))
            strides.unsafe_set(result_axis, self.unsafe_stride_at(source_axis))
        return Self(shape, strides, self.offset)

    def transpose(self) raises -> Self:
        """Return a layout with its axis order reversed.

        Returns:
            A layout with reversed shape and strides.

        Examples:
            ```mojo
            var transposed = NDLayout([2, 3]).transpose()
            ```
        """
        var axes = List[Int](capacity=len(self))
        for axis in range(len(self)):
            axes.append(len(self) - 1 - axis)
        return self.transpose(axes)

    def reshape(self, shape: Shape) raises -> Self:
        """Return a C-contiguous layout with a new compatible shape.

        One dimension may be `-1` and is inferred from the current element
        count. The source layout must be C contiguous.

        Args:
            shape: Requested dimensions.

        Returns:
            A reshaped layout preserving the source offset.

        Raises:
            NumojoError: If the source is non-contiguous or shape is invalid.

        Examples:
            ```mojo
            var reshaped = NDLayout([2, 3]).reshape(Shape(3, 2))
            ```
        """
        if not self.is_c_contiguous():
            raise Error(
                NumojoError(
                    category="shape",
                    message="Reshape requires a C-contiguous layout.",
                    location="NDLayout.reshape(shape)",
                )
            )

        var resolved = shape.tolist()
        var inferred_axis = -1
        for axis in range(len(resolved)):
            if resolved[axis] == -1:
                if inferred_axis != -1:
                    raise Error(
                        NumojoError(
                            category="shape",
                            message="Only one inferred dimension is allowed.",
                            location="NDLayout.reshape(shape)",
                        )
                    )
                inferred_axis = axis
                resolved[axis] = 1

        var known_size = Self._list_product(resolved)
        if inferred_axis != -1:
            if known_size == 0 or self.size % known_size != 0:
                raise Error(
                    NumojoError(
                        category="shape",
                        message="Cannot infer a compatible reshape dimension.",
                        location="NDLayout.reshape(shape)",
                    )
                )
            resolved[inferred_axis] = self.size // known_size

        var result = Self(resolved)
        if result.size != self.size:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Cannot reshape {} elements into shape {}."
                    ).format(self.size, shape.__str__()),
                    location="NDLayout.reshape(shape)",
                )
            )
        result.offset = self.offset
        return result^

    def reshape(self, shape: List[Int]) raises -> Self:
        """Return a C-contiguous layout with new list-provided dimensions.

        Args:
            shape: Requested dimensions, optionally containing one `-1`.

        Returns:
            A reshaped layout preserving the source offset.

        Raises:
            NumojoError: If the source is non-contiguous or shape is invalid.

        Examples:
            ```mojo
            var reshaped = NDLayout([2, 3]).reshape([3, 2])
            ```
        """
        return self.reshape(Shape(shape))

    def sliced(self, first: LayoutIndex, *rest: LayoutIndex) raises -> Self:
        """Apply mixed integer, slice, ellipsis, and new-axis indices.

        Args:
            first: First required index.
            rest: Remaining indices.

        Returns:
            A layout describing the indexed view.

        Raises:
            NumojoError: If indices are invalid for the layout.

        Examples:
            ```mojo
            var result = NDLayout([2, 3]).sliced(1, Slice(None, None, -1))
            ```
        """
        var consumed = 0
        var ellipses = 0
        for index_number in range(len(rest) + 1):
            var index = first if index_number == 0 else rest[index_number - 1]
            if index.value.isa[EllipsisType]():
                ellipses += 1
            elif not index.value.isa[NewAxis]():
                consumed += 1

        if ellipses > 1:
            raise Error(
                NumojoError(
                    category="index",
                    message="Only one ellipsis is allowed.",
                    location="NDLayout.sliced(first, *rest)",
                )
            )
        if consumed > len(self):
            raise Error(
                NumojoError(
                    category="index",
                    message="Too many indices for the layout rank.",
                    location="NDLayout.sliced(first, *rest)",
                )
            )

        var result = self.copy()
        var axis = 0
        for index_number in range(len(rest) + 1):
            var index = first if index_number == 0 else rest[index_number - 1]
            if index.value.isa[EllipsisType]():
                axis += len(self) - consumed
            elif index.value.isa[NewAxis]():
                var shape = List[Int](capacity=len(result) + 1)
                var strides = List[Int](capacity=len(result) + 1)
                for result_axis in range(len(result) + 1):
                    if result_axis == axis:
                        shape.append(1)
                        strides.append(0)
                    if result_axis < len(result):
                        shape.append(result.unsafe_shape_at(result_axis))
                        strides.append(result.unsafe_stride_at(result_axis))
                result = Self(shape, strides, result.offset)
                axis += 1
            elif index.value.isa[Int]():
                result = result.select(axis, index.value[Int])
            else:
                var slice_value = index.value[Slice]
                var step = slice_value.step.or_else(1)
                if step == 0 or step == Int.MIN:
                    raise Error(
                        NumojoError(
                            category="value",
                            message="Slice step must be nonzero.",
                            location="NDLayout.sliced(first, *rest)",
                        )
                    )
                var start, stop, increment = slice_value.indices(
                    result.unsafe_shape_at(axis)
                )
                var count = 0
                if increment > 0 and start < stop:
                    count = 1 + (stop - start - 1) // increment
                elif increment < 0 and start > stop:
                    count = 1 + (start - stop - 1) // (-increment)
                result = result.slice(axis, start, count, increment)
                axis += 1
        return result^

    @always_inline("nodebug")
    def shape_at(self, axis: Int) raises -> Int:
        """Return the extent of one axis without allocating.

        Args:
            axis: Dimension index.

        Returns:
            The extent of `axis`.

        Examples:
            ```mojo
            var extent = NDLayout([2, 3]).shape_at(1)
            ```
        """
        return self._shape[axis]

    @always_inline("nodebug")
    def unsafe_shape_at(self, axis: Int) -> Int:
        """Return one extent without bounds checking.

        Args:
            axis: Dimension index known to be in bounds.

        Returns:
            The extent of `axis`.

        Examples:
            ```mojo
            var extent = NDLayout([2, 3]).unsafe_shape_at(1)
            ```
        """
        return self._shape.unsafe_get(axis)

    @always_inline("nodebug")
    def stride_at(self, axis: Int) raises -> Int:
        """Return the stride of one axis without allocating.

        Args:
            axis: Dimension index.

        Returns:
            The element stride of `axis`.

        Examples:
            ```mojo
            var stride = NDLayout([2, 3]).stride_at(0)
            ```
        """
        return self._strides[axis]

    @always_inline("nodebug")
    def unsafe_stride_at(self, axis: Int) -> Int:
        """Return one stride without bounds checking.

        Args:
            axis: Dimension index known to be in bounds.

        Returns:
            The element stride of `axis`.

        Examples:
            ```mojo
            var stride = NDLayout([2, 3]).unsafe_stride_at(0)
            ```
        """
        return self._strides.unsafe_get(axis)

    def shape(self) -> Shape:
        """Return a copy of the shape.

        Returns:
            A copy of the dimensions.

        Examples:
            ```mojo
            var shape = NDLayout([2, 3]).shape()
            ```
        """
        return self._shape.copy()

    def strides(self) -> Strides:
        """Return a copy of the strides.

        Returns:
            A copy of the element strides.

        Examples:
            ```mojo
            var strides = NDLayout([2, 3]).strides()
            ```
        """
        return self._strides.copy()

    def shape_list(self) -> List[Int]:
        """Return the shape dimensions as a list.

        Returns:
            A list containing the dimensions.

        Examples:
            ```mojo
            var shape = NDLayout([2, 3]).shape_list()
            ```
        """
        return self._shape.tolist()

    def strides_list(self) -> List[Int]:
        """Return the strides as a list.

        Returns:
            A list containing the element strides.

        Examples:
            ```mojo
            var strides = NDLayout([2, 3]).strides_list()
            ```
        """
        return self._strides.tolist()

    def is_c_contiguous(self) -> Bool:
        """Return whether the layout is C contiguous.

        Returns:
            True when the metadata describes C-contiguous storage.

        Examples:
            ```mojo
            var contiguous = NDLayout([2, 3]).is_c_contiguous()
            ```
        """
        if len(self) == 0 or self.size == 0:
            return True
        var expected = 1
        for axis in range(len(self) - 1, -1, -1):
            var extent = self._shape.unsafe_get(axis)
            if extent > 1:
                if self._strides.unsafe_get(axis) != expected:
                    return False
                expected *= extent
        return True

    @always_inline("nodebug")
    def contiguous(self) -> Bool:
        """Return whether the layout is C contiguous.

        Returns:
            True when logical flat order matches contiguous storage order.

        Examples:
            ```mojo
            var contiguous = NDLayout([2, 3]).contiguous()
            ```
        """
        return self.is_c_contiguous()

    def is_f_contiguous(self) -> Bool:
        """Return whether the layout is Fortran contiguous.

        Returns:
            True when the metadata describes Fortran-contiguous storage.

        Examples:
            ```mojo
            var contiguous = NDLayout([2, 3], order="F").is_f_contiguous()
            ```
        """
        if len(self) == 0 or self.size == 0:
            return True
        var expected = 1
        for axis in range(len(self)):
            var extent = self._shape.unsafe_get(axis)
            if extent > 1:
                if self._strides.unsafe_get(axis) != expected:
                    return False
                expected *= extent
        return True

    def __str__(self) -> String:
        """Return a compact representation of the layout."""
        var result = String("NDLayout(shape=(")
        for i in range(len(self)):
            if i > 0:
                result += ", "
            result += String(self._shape.unsafe_get(i))
        result += "), strides=("
        for i in range(len(self)):
            if i > 0:
                result += ", "
            result += String(self._strides.unsafe_get(i))
        return (
            result
            + "), offset="
            + String(self.offset)
            + ", size="
            + String(self.size)
            + ")"
        )

    def write_to[W: Writer](self, mut writer: W):
        """Write the layout representation to a writer.

        Parameters:
            W: Writer type.

        Args:
            writer: Destination writer.

        Examples:
            ```mojo
            print(NDLayout([2, 3]))
            ```
        """
        writer.write(self.__str__())
