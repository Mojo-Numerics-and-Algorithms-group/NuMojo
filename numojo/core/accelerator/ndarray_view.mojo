# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator NDArray view
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator NDArray view (numojo.core.accelerator.ndarray_view).
================================================================
Borrowed N-dimensional views over accelerator-array storage.

Exports
-------
- `AcceleratorNDArrayView`: Origin-preserving immutable or mutable view.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.array_like import NDArrayLike
from numojo.core.accelerator.device import Device
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.layout.flags import Flags
from numojo.core.layout.ndlayout import NDLayout, Shape, Strides


struct AcceleratorNDArrayView[
    mut: Bool,
    //,
    dtype: DType,
    device: Device,
    origin: Origin[mut=mut],
](Movable, NDArrayLike, Sized):
    """Borrowed N-dimensional array view.

    Parameters:
        mut: Whether the view can mutate its borrowed storage.
        dtype: Element dtype.
        device: Device holding the borrowed storage.
        origin: Origin of the owner storage borrow.
    """

    comptime ElementType = Self.dtype
    comptime DeviceType = Self.device
    comptime Writable = Self.mut

    var _ptr: Pointer[Scalar[Self.dtype], Self.origin]
    var _layout: NDLayout
    var flags: Flags

    # ===----------------------------------------------------------------------=== #
    # Lifecycle and borrowing
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], Self.origin],
        var layout: NDLayout,
    ):
        """Create a view over borrowed storage.

        Args:
            ptr: Pointer to the start of the owner's storage allocation.
            layout: Logical layout within that allocation.

        Examples:
            Views are normally constructed with `array.view()` or
            `array.view_mut()`.
        """
        self._ptr = ptr
        self._layout = layout^
        self.flags = Flags(
            c_contiguous=self._layout.is_c_contiguous(),
            f_contiguous=self._layout.is_f_contiguous(),
            owndata=False,
            writeable=Self.mut,
        )

    @always_inline("nodebug")
    def view(
        self,
    ) -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(Self.origin),
    ]:
        """Create an immutable reborrow of this view.

        Returns:
            Immutable view preserving the owner storage origin.

        Examples:
            `var read_only = writable_view.view()` creates an immutable view.
        """
        return AcceleratorNDArrayView[
            mut=False,
            dtype=Self.dtype,
            device=Self.device,
            origin=ImmOrigin(Self.origin),
        ](
            self._ptr.as_imm(),
            self._layout.copy(),
        )

    @always_inline("nodebug")
    def view_mut(mut self) -> Self where Self.mut:
        """Create a mutable reborrow of this writable view.

        Returns:
            Writable view preserving the owner storage origin.

        Examples:
            `var writable = view.view_mut()` reborrows a writable view.
        """
        return Self(
            self._ptr,
            self._layout.copy(),
        )

    # ===----------------------------------------------------------------------=== #
    # Pointer access
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def unsafe_ptr[
        o: Origin
    ](ref[o] self,) -> Pointer[Scalar[Self.dtype], ImmOrigin(o)]:
        """Return an immutable pointer tied to this view borrow.

        Parameters:
            o: Origin of the current view borrow.

        Returns:
            Immutable pointer to the first logical element.

        Examples:
            Use `view.unsafe_ptr()` for a SIMD loop or kernel launch.
        """
        return (
            self._ptr.as_imm()
            .unsafe_origin_cast[ImmOrigin(o)]()
            .unsafe_offset(self._layout.offset)
        )

    @always_inline("nodebug")
    def unsafe_mut_ptr[
        o: MutOrigin
    ](ref[o] self,) -> Pointer[Scalar[Self.dtype], o]:
        """Return a mutable pointer tied to this writable view borrow.

        Parameters:
            o: Mutable origin of the view borrow.

        Returns:
            Mutable pointer to the first logical element.

        Examples:
            Use `view.unsafe_mut_ptr()` for generic output algorithms.
        """
        comptime assert Self.Writable, "cannot mutate a read-only view"
        var offset = self._layout.offset
        return (
            self._ptr.unsafe_mut_cast[True]()
            .unsafe_origin_cast[o]()
            .unsafe_offset(offset)
        )

    # ===----------------------------------------------------------------------=== #
    # Layout metadata
    # ===----------------------------------------------------------------------=== #

    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied layout metadata.

        Examples:
            Use `view.layout()` in generic indexing code.
        """
        return self._layout.copy()

    @always_inline("nodebug")
    def ndim(self) -> Int:
        """Return the number of dimensions.

        Returns:
            Logical array rank.

        Examples:
            `view.ndim()` returns zero for a scalar view.
        """
        return len(self._layout)

    @always_inline("nodebug")
    def __len__(self) -> Int:
        if self.ndim() == 0:
            return 1
        return self._layout.unsafe_shape_at(0)

    def shape(self) -> Shape:
        """Return a copy of the array shape.

        Returns:
            Copied shape metadata.

        Examples:
            Use `view.shape()` to inspect logical dimensions.
        """
        return self._layout.shape()

    def strides(self) -> Strides:
        """Return a copy of the element strides.

        Returns:
            Copied element strides.

        Examples:
            Use `view.strides()` to inspect logical traversal.
        """
        return self._layout.strides()

    @always_inline("nodebug")
    def size(self) -> Int:
        """Return the number of logical elements.

        Returns:
            Product of all logical dimensions.

        Examples:
            `view.size()` returns one for a scalar view.
        """
        return self._layout.size

    @always_inline("nodebug")
    def offset(self) -> Int:
        """Return the logical origin's storage offset.

        Returns:
            Element offset within the owner allocation.

        Examples:
            A selected row can have a nonzero `view.offset()`.
        """
        return self._layout.offset

    @always_inline("nodebug")
    def is_c_contiguous(self) -> Bool:
        """Return whether the view is C contiguous.

        Returns:
            C-contiguity of the logical layout.

        Examples:
            Check `view.is_c_contiguous()` before flat pointer traversal.
        """
        return self._layout.is_c_contiguous()

    @always_inline("nodebug")
    def is_f_contiguous(self) -> Bool:
        """Return whether the view is Fortran contiguous.

        Returns:
            Fortran-contiguity of the logical layout.

        Examples:
            Use `view.is_f_contiguous()` to select traversal order.
        """
        return self._layout.is_f_contiguous()

    @always_inline("nodebug")
    def owns_data(self) -> Bool:
        """Return whether the view owns its backing allocation.

        Returns:
            Always `False` for a view.

        Examples:
            `view.owns_data()` is false.
        """
        return False

    @always_inline("nodebug")
    def is_writeable(self) -> Bool:
        """Return whether this view has mutable storage access.

        Returns:
            The compile-time view mutability as a runtime boolean.

        Examples:
            `array.view_mut().is_writeable()` is true.
        """
        return Self.mut

    @always_inline("nodebug")
    def is_cpu(self) -> Bool:
        """Return whether the view refers to CPU storage.

        Returns:
            `True` for `Device.CPU`.

        Examples:
            Use `view.is_cpu()` before host scalar access.
        """
        return Self.device.type == "cpu"

    @always_inline("nodebug")
    def is_gpu(self) -> Bool:
        """Return whether the view refers to GPU storage.

        Returns:
            `True` for a GPU device.

        Examples:
            Use `view.is_gpu()` before a kernel launch.
        """
        return Self.device.type == "gpu"

    @always_inline("nodebug")
    def num_elements(self) -> Int:
        """Return the logical element count.

        Returns:
            Same value as `size()`.

        Examples:
            Use `view.num_elements()` in generic adapters.
        """
        return self.size()

    # ===----------------------------------------------------------------------=== #
    # Layout views
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def _with_layout(self, var layout: NDLayout) -> Self:
        return Self(
            self._ptr,
            layout^,
        )

    def slice(
        self,
        axis: Int,
        start: Int,
        count: Int,
        step: Int = 1,
    ) raises -> Self:
        """Create a sliced view.

        Args:
            axis: Axis to slice.
            start: First selected index.
            count: Number of selected elements.
            step: Distance between selected elements.

        Returns:
            Sliced view preserving the owner storage origin.

        Examples:
            `view.slice(0, 1, 2)` selects two entries along axis zero.
        """
        return self._with_layout(self._layout.slice(axis, start, count, step))

    def select(self, axis: Int, index: Int) raises -> Self:
        """Create a view with one indexed axis removed.

        Args:
            axis: Axis to remove.
            index: Coordinate selected from that axis.

        Returns:
            Rank-one-lower view preserving storage provenance.

        Examples:
            `view.select(0, 1)` selects the second row.
        """
        return self._with_layout(self._layout.select(axis, index))

    def transpose(self, axes: List[Int]) raises -> Self:
        """Create a view with permuted axes.

        Args:
            axes: Complete axis permutation.

        Returns:
            View with shape and strides permuted together.

        Examples:
            `view.transpose([1, 0])` swaps matrix axes.
        """
        return self._with_layout(self._layout.transpose(axes))

    def transpose(self) raises -> Self:
        """Create a view with reversed axes.

        Returns:
            View with axes in reverse order.

        Examples:
            `view.transpose()` transposes a matrix.
        """
        return self._with_layout(self._layout.transpose())

    def reshape(self, shape: Shape) raises -> Self:
        """Create a reshaped view.

        Args:
            shape: Requested compatible shape.

        Returns:
            Reshaped view preserving storage provenance.

        Examples:
            `view.reshape(Shape(2, 3))` creates a two-dimensional view.
        """
        return self._with_layout(self._layout.reshape(shape))

    def reshape(self, shape: List[Int]) raises -> Self:
        """Create a reshaped view from list dimensions.

        Args:
            shape: Requested compatible dimensions.

        Returns:
            Reshaped view preserving storage provenance.

        Examples:
            `view.reshape([2, 3])` creates a two-dimensional view.
        """
        return self._with_layout(self._layout.reshape(shape))

    # ===----------------------------------------------------------------------=== #
    # Scalar access
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def _normalize(self, index: Int, dim: Int) -> Int:
        return index if index >= 0 else index + dim

    def item(self, flat_index: Int) raises -> Scalar[Self.dtype]:
        """Return one element at a logical flat index.

        Args:
            flat_index: Logical flat index.

        Returns:
            Selected scalar value.

        Raises:
            NumojoError: If host access is requested for a GPU view.

        Examples:
            `view.item(0)` reads the first logical element.
        """
        comptime if Self.device.type == "cpu":
            return self._ptr[
                unsafe_offset=self._layout.flat_position(flat_index)
            ]
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side item access is not available for GPU"
                        " views. Materialize on the host first."
                    ),
                    location="AcceleratorNDArrayView.item(flat_index)",
                )
            )

    def item(self, *indices: Int) raises -> Scalar[Self.dtype]:
        """Return one element at N-dimensional coordinates.

        Args:
            indices: One coordinate per axis.

        Returns:
            Selected scalar value.

        Raises:
            NumojoError: If coordinate count or bounds are invalid, or host
                access is requested for a GPU view.

        Examples:
            `view.item(1, 2)` reads one matrix element.
        """
        if len(indices) != self.ndim():
            raise Error(
                NumojoError(
                    category="index",
                    message=String("Expected {} indices but got {}.").format(
                        self.ndim(), len(indices)
                    ),
                    location="AcceleratorNDArrayView.item(*indices)",
                )
            )
        var storage_offset = self._layout.offset
        for axis in range(self.ndim()):
            var extent = self._layout.shape_at(axis)
            var normalized = self._normalize(indices[axis], extent)
            if normalized < 0 or normalized >= extent:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for axis {} with size {}."
                        ).format(indices[axis], axis, extent),
                        location="AcceleratorNDArrayView.item(*indices)",
                    )
                )
            storage_offset += normalized * self._layout.stride_at(axis)
        comptime if Self.device.type == "cpu":
            return self._ptr[unsafe_offset=storage_offset]
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side item access is not available for GPU"
                        " views. Materialize on the host first."
                    ),
                    location="AcceleratorNDArrayView.item(*indices)",
                )
            )

    def itemset(
        mut self,
        flat_index: Int,
        value: Scalar[Self.dtype],
    ) raises where Self.mut:
        """Set one element at a logical flat index.

        Args:
            flat_index: Logical flat index.
            value: Scalar value to store.

        Raises:
            NumojoError: If host access is requested for a GPU view.

        Examples:
            `view.itemset(0, value)` writes the first logical element.
        """
        comptime if Self.device.type == "cpu":
            comptime if Self.mut:
                var ptr = self._ptr.unsafe_mut_cast[True]()
                ptr[
                    unsafe_offset=self._layout.flat_position(flat_index)
                ] = value
            else:
                pass
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side itemset is not available for GPU"
                        " views."
                    ),
                    location="AcceleratorNDArrayView.itemset",
                )
            )

    def __getitem__(self) raises -> Scalar[Self.dtype]:
        if self.ndim() != 0 and self.size() != 1:
            raise Error(
                NumojoError(
                    category="index",
                    message="Use `a[]` only for 0-D or size-1 arrays.",
                    location="AcceleratorNDArrayView.__getitem__()",
                )
            )
        return self.item(0)

    def __getitem__(self, index: Item) raises -> Scalar[Self.dtype]:
        if len(index) != self.ndim():
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices in Item but got {}."
                    ).format(self.ndim(), len(index)),
                    location="AcceleratorNDArrayView.__getitem__(Item)",
                )
            )
        var storage_offset = self._layout.offset
        for axis in range(self.ndim()):
            var extent = self._layout.shape_at(axis)
            var normalized = self._normalize(index[axis], extent)
            if normalized < 0 or normalized >= extent:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for axis {} with size {}."
                        ).format(index[axis], axis, extent),
                        location="AcceleratorNDArrayView.__getitem__(Item)",
                    )
                )
            storage_offset += normalized * self._layout.stride_at(axis)
        comptime if Self.device.type == "cpu":
            return self._ptr[unsafe_offset=storage_offset]
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side item access is not available for GPU"
                        " views. Materialize on the host first."
                    ),
                    location="AcceleratorNDArrayView.__getitem__(Item)",
                )
            )

    def __getitem__(self, index: Int) raises -> Self:
        return self.select(0, index)
