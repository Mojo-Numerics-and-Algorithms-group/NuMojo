# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator NDArray
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
AcceleratorNDArray (numojo.core.accelerator_ndarray).
=====================================================
Device-aware NDArray with accelerator support.

Device-aware NDArray that stores data in AcceleratorDataContainer with
GPU acceleration support.

Exports
-------
- `AcceleratorNDArray`: Device-aware N-dimensional array type.
- `AcceleratorNDArrayView`: Borrowed device-aware N-dimensional array view.
- `NDArrayLike`: Common read-only array interface.
"""

# ===----------------------------------------------------------------------=== #
# External
# ===----------------------------------------------------------------------=== #
from max.gpu.host import DeviceContext

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.array_like import NDArrayLike
from numojo.core.accelerator.device import Device
import numojo.core.accelerator.kernels as kernels
from numojo.core.accelerator.ndarray_view import AcceleratorNDArrayView
from numojo.core.dtype.default_dtype import _concise_dtype_str
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.layout.flags import Flags
from numojo.core.layout.ndlayout import NDLayout, Shape, Strides
from numojo.core.memory.storage import AcceleratorDataContainer
from numojo.core.ndarray import NDArray


struct AcceleratorNDArray[
    dtype: DType = DType.float64, device: Device = Device.CPU
](
    Copyable,
    Movable,
    NDArrayLike,
    Sized,
    Writable,
):
    """Device-aware N-dimensional array.

    Parameters:
        dtype: Element dtype.
        device: Target device (`Device.CPU`, `Device.CUDA`, `Device.ROCM`,
            `Device.MPS`).
    """

    var _buf: AcceleratorDataContainer[Self.dtype, Self.device]
    var _layout: NDLayout
    var flags: Flags

    comptime ElementType = Self.dtype
    comptime DeviceType = Self.device
    comptime Writable = True

    # ===------------------------------------------------------------------=== #
    # Lifecycle
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __init__(out self):
        self._buf = AcceleratorDataContainer[Self.dtype, Self.device]()
        self._layout = NDLayout()
        self.flags = Flags(
            c_contiguous=True,
            f_contiguous=True,
            owndata=False,
            writeable=False,
        )

    @always_inline("nodebug")
    def __init__(out self, shape: Shape, order: String = "C") raises:
        self._layout = NDLayout(shape, order=order)
        self._buf = AcceleratorDataContainer[Self.dtype, Self.device](
            self._layout.size
        )
        self.flags = Flags(
            c_contiguous=self._layout.is_c_contiguous(),
            f_contiguous=self._layout.is_f_contiguous(),
            owndata=True,
            writeable=True,
        )

    @always_inline("nodebug")
    def __init__(out self, shape: List[Int], order: String = "C") raises:
        self = Self(shape=Shape(shape), order=order)

    @always_inline("nodebug")
    def __init__(out self, *shape: Int, order: String = "C") raises:
        var dimensions = List[Int](capacity=len(shape))
        for dimension in shape:
            dimensions.append(dimension)
        self = Self(shape=Shape(dimensions), order=order)

    @always_inline("nodebug")
    def __init__(
        out self,
        shape: Shape,
        strides: Strides,
        offset: Int,
        flags: Flags,
    ) raises:
        self._layout = NDLayout(shape, strides, offset)
        self._buf = AcceleratorDataContainer[Self.dtype, Self.device](
            self._layout.size
        )
        self.flags = flags

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        self._layout = copy._layout.copy()
        self.flags = copy.flags
        self._buf = copy._buf.copy()

    @always_inline("nodebug")
    def __init__(out self, *, deinit move: Self):
        self._layout = move._layout^
        self.flags = move.flags
        self._buf = move._buf^

    @always_inline("nodebug")
    def view[
        o: Origin
    ](
        ref[o] self,
    ) -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create an immutable borrowed view of this array.

        Parameters:
            o: Origin of the current owner borrow.

        Returns:
            A view whose storage origin is tied to this borrow.

        Examples:
            ```mojo
            var shared = array.view()
            ```
        """
        return AcceleratorNDArrayView[
            mut=False,
            dtype=Self.dtype,
            device=Self.device,
            origin=ImmOrigin(o),
        ](
            self._unsafe_base_ptr().as_imm(),
            self._layout.copy(),
        )

    @always_inline("nodebug")
    def view_mut[
        o: MutOrigin
    ](
        ref[o] self,
    ) -> AcceleratorNDArrayView[
        mut=True, dtype=Self.dtype, device=Self.device, origin=o
    ]:
        """Create a mutable borrowed view of this array.

        Parameters:
            o: Mutable origin of the current owner borrow.

        Returns:
            A writable view whose storage origin is tied to this mutable borrow.

        Examples:
            `var writable = array.view_mut()` creates a writable view.
        """
        return AcceleratorNDArrayView[
            mut=True, dtype=Self.dtype, device=Self.device, origin=o
        ](
            self._unsafe_base_ptr(),
            self._layout.copy(),
        )

    def slice[
        o: Origin
    ](
        ref[o] self,
        axis: Int,
        start: Int,
        count: Int,
        step: Int = 1,
    ) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create a view sliced along one axis.

        Parameters:
            o: Origin of the current owner borrow.

        Args:
            axis: Axis to slice.
            start: First selected index.
            count: Number of selected elements.
            step: Distance between selected elements.

        Returns:
            A view with updated layout metadata and shared storage.

        Examples:
            ```mojo
            var every_other = array.slice(0, 0, 3, 2)
            ```
        """
        return self._make_view(self._layout.slice(axis, start, count, step))

    def select[
        o: Origin
    ](ref[o] self, axis: Int, index: Int) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create a view with one indexed axis removed.

        Parameters:
            o: Origin of the current owner borrow.

        Args:
            axis: Axis to remove.
            index: Element selected from that axis.

        Returns:
            A rank-one-lower view sharing the source storage.

        Examples:
            ```mojo
            var row = array.select(0, 1)
            ```
        """
        return self._make_view(self._layout.select(axis, index))

    def transpose[
        o: Origin
    ](ref[o] self, axes: List[Int]) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create a view with axes in the requested order.

        Parameters:
            o: Origin of the current owner borrow.

        Args:
            axes: Complete axis permutation.

        Returns:
            A view with shape and strides permuted together.

        Examples:
            ```mojo
            var permuted = array.transpose([1, 0])
            ```
        """
        return self._make_view(self._layout.transpose(axes))

    def transpose[
        o: Origin
    ](ref[o] self) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create a view with reversed axis order.

        Parameters:
            o: Origin of the current owner borrow.

        Returns:
            A view with reversed shape and strides.

        Examples:
            ```mojo
            var transposed = array.transpose()
            ```
        """
        return self._make_view(self._layout.transpose())

    def reshape[
        o: Origin
    ](ref[o] self, shape: Shape) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create a C-contiguous view with a compatible shape.

        Parameters:
            o: Origin of the current owner borrow.

        Args:
            shape: Requested shape, optionally containing one `-1`.

        Returns:
            A reshaped view sharing the source storage.

        Examples:
            ```mojo
            var reshaped = array.reshape(Shape(3, 2))
            ```
        """
        return self._make_view(self._layout.reshape(shape))

    def reshape[
        o: Origin
    ](ref[o] self, shape: List[Int]) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        """Create a C-contiguous view with list-provided dimensions.

        Parameters:
            o: Origin of the current owner borrow.

        Args:
            shape: Requested dimensions, optionally containing one `-1`.

        Returns:
            A reshaped view sharing the source storage.

        Examples:
            ```mojo
            var reshaped = array.reshape([3, 2])
            ```
        """
        return self._make_view(self._layout.reshape(shape))

    @always_inline("nodebug")
    def _make_view[
        o: Origin
    ](
        ref[o] self,
        var layout: NDLayout,
    ) -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        return AcceleratorNDArrayView[
            mut=False,
            dtype=Self.dtype,
            device=Self.device,
            origin=ImmOrigin(o),
        ](
            self._unsafe_base_ptr().as_imm(),
            layout^,
        )

    # ===----------------------------------------------------------------------=== #
    # Layout metadata
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def ndim(self) -> Int:
        """Return the number of dimensions.

        Returns:
            The array rank.

        Examples:
            ```mojo
            var rank = array.ndim()
            ```
        """
        return len(self._layout)

    def shape(self) -> Shape:
        """Return the array shape.

        Returns:
            A copy of the array dimensions.

        Examples:
            ```mojo
            var shape = array.shape()
            ```
        """
        return self._layout.shape()

    def strides(self) -> Strides:
        """Return the element strides.

        Returns:
            A copy of the array's element strides.

        Examples:
            ```mojo
            var strides = array.strides()
            ```
        """
        return self._layout.strides()

    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied layout metadata.

        Examples:
            Use `array.layout()` in generic indexing code.
        """
        return self._layout.copy()

    @always_inline("nodebug")
    def size(self) -> Int:
        """Return the number of logical elements.

        Returns:
            The product of the array dimensions.

        Examples:
            ```mojo
            var element_count = array.size()
            ```
        """
        return self._layout.size

    @always_inline("nodebug")
    def offset(self) -> Int:
        """Return the first element's offset in the underlying storage.

        Returns:
            The element offset of the logical array origin.

        Examples:
            ```mojo
            var first_element = array.offset()
            ```
        """
        return self._layout.offset

    # ===------------------------------------------------------------------=== #
    # Representation
    # ===------------------------------------------------------------------=== #

    def _as_cpu_ndarray_for_display(self) raises -> NDArray[Self.dtype]:
        var shape = self._layout.shape_list()
        var contiguous_layout = NDLayout(shape)
        var out = NDArray[Self.dtype](
            shape, contiguous_layout.strides_list(), 0
        )
        for i in range(self.size()):
            out.itemset(i, self.item(i))
        return out^

    def __str__(self) -> String:
        var res: String
        try:
            if self.ndim() == 0:
                res = (
                    String(self.item(0))
                    + "  (0darray["
                    + _concise_dtype_str(self.dtype)
                    + "], use `[]` or `.item()` to unpack)"
                )
            else:
                comptime if Self.device.type == "cpu":
                    res = self._as_cpu_ndarray_for_display().__str__()
                else:
                    var host = self.to_host()
                    res = host._as_cpu_ndarray_for_display().__str__()
                var order = String("non-contiguous")
                if self.flags.C_CONTIGUOUS:
                    order = "C"
                elif self.flags.F_CONTIGUOUS:
                    order = "F"
                res += (
                    "\n"
                    + String(self.ndim())
                    + "D-array  Shape: "
                    + self.shape().__str__()
                    + "  Strides: "
                    + self.strides().__str__()
                    + "  DType: "
                    + _concise_dtype_str(self.dtype)
                    + "  order: "
                    + order
                    + "  own data: "
                    + String(self.flags.OWNDATA)
                    + "  device: "
                    + Self.device.device_name()
                )
        except e:
            res = String("Cannot convert array to string.\n") + String(e)
        return res

    def __repr__(self) -> String:
        return self.__str__()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    def __len__(self) -> Int:
        if self.ndim() == 0:
            return 1
        return self._layout.unsafe_shape_at(0)

    # ===------------------------------------------------------------------=== #
    # Device helpers
    # ===------------------------------------------------------------------=== #

    def is_cpu(self) -> Bool:
        return Self.device.type == "cpu"

    def is_gpu(self) -> Bool:
        return Self.device.type == "gpu"

    @always_inline("nodebug")
    def is_c_contiguous(self) -> Bool:
        """Return whether the array is C contiguous.

        Returns:
            C-contiguity of the logical layout.

        Examples:
            Check `array.is_c_contiguous()` before flat pointer traversal.
        """
        return self._layout.is_c_contiguous()

    @always_inline("nodebug")
    def is_f_contiguous(self) -> Bool:
        """Return whether the array is Fortran contiguous.

        Returns:
            Fortran-contiguity of the logical layout.

        Examples:
            Use `array.is_f_contiguous()` to select traversal order.
        """
        return self._layout.is_f_contiguous()

    @always_inline("nodebug")
    def owns_data(self) -> Bool:
        """Return whether this array owns its backing allocation.

        Returns:
            Ownership flag for the backing storage.

        Examples:
            `array.owns_data()` is true for a normally constructed owner.
        """
        return self.flags.OWNDATA

    @always_inline("nodebug")
    def is_writeable(self) -> Bool:
        """Return whether this array exposes mutable storage.

        Returns:
            Runtime writeability flag.

        Examples:
            Use `array.is_writeable()` for validation.
        """
        return self.flags.WRITEABLE

    @always_inline("nodebug")
    def _unsafe_base_ptr[
        o: Origin
    ](ref[o] self,) -> Pointer[Scalar[Self.dtype], o]:
        comptime if Self.device.type == "cpu":
            return (
                self._buf.host_storage.unsafe_value()
                .ptr.unsafe_mut_cast[o.mut]()
                .unsafe_origin_cast[o]()
            )
        else:
            return (
                self._buf.device_storage.unsafe_value()
                .unsafe_ptr()
                .unsafe_mut_cast[o.mut]()
                .unsafe_origin_cast[o]()
            )

    @always_inline("nodebug")
    def unsafe_ptr[
        o: Origin
    ](ref[o] self,) -> Pointer[Scalar[Self.dtype], ImmOrigin(o)]:
        """Return an immutable pointer to the first logical element.

        Parameters:
            o: Origin of the current owner borrow.

        Returns:
            Immutable pointer tied to the current owner borrow.

        Examples:
            Use `array.unsafe_ptr()` for a SIMD loop or kernel launch.
        """
        var offset = self._layout.offset
        return self._unsafe_base_ptr().as_imm().unsafe_offset(offset)

    @always_inline("nodebug")
    def unsafe_mut_ptr[
        o: MutOrigin
    ](ref[o] self,) -> Pointer[Scalar[Self.dtype], o]:
        """Return a mutable pointer to the first logical element.

        Parameters:
            o: Mutable origin of the current owner borrow.

        Returns:
            Mutable pointer tied to the current mutable owner borrow.

        Examples:
            Use `array.unsafe_mut_ptr()` for output SIMD loops.
        """
        comptime assert Self.Writable, "cannot mutate a read-only array"
        var offset = self._layout.offset
        return self._unsafe_base_ptr().unsafe_offset(offset)

    def device_context(
        self,
    ) -> DeviceContext where Self.device.type == "gpu":
        """Return the `DeviceContext` backing this array's GPU storage."""
        return self._buf.device_storage.unsafe_value().handle.device_context()

    def num_elements(self) -> Int:
        return self.size()

    # ===------------------------------------------------------------------=== #
    # Indexing and scalar access
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def normalize(self, index: Int, dim: Int) -> Int:
        return index if index >= 0 else index + dim

    @always_inline("nodebug")
    def _coordinates_offset(self, coordinates: Item) raises -> Int:
        var result = self.offset()
        for axis in range(self.ndim()):
            result += coordinates[axis] * self._layout.stride_at(axis)
        return result

    @always_inline("nodebug")
    def _flat_offset(self, flat_index: Int) raises -> Int:
        return self._layout.flat_position(flat_index)

    def item(self, flat_index: Int) raises -> Scalar[Self.dtype]:
        comptime if Self.device.type == "cpu":
            var off = self._flat_offset(flat_index)
            return self._buf[off]
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side item access is not available for GPU"
                        " arrays. Call `.to_host()` first."
                    ),
                    location="AcceleratorNDArray.item(flat_index)",
                )
            )

    def item(self, *indices: Int) raises -> Scalar[Self.dtype]:
        if len(indices) != self.ndim():
            raise Error(
                NumojoError(
                    category="index",
                    message=String("Expected {} indices but got {}.").format(
                        self.ndim(), len(indices)
                    ),
                    location="AcceleratorNDArray.item(*indices)",
                )
            )

        var coords = Item(ndim=self.ndim())
        for i in range(self.ndim()):
            var extent = self._layout.shape_at(i)
            var n = self.normalize(indices[i], extent)
            if n < 0 or n >= extent:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for axis {} with size {}."
                        ).format(indices[i], i, extent),
                        location="AcceleratorNDArray.item(*indices)",
                    )
                )
            coords[i] = n

        comptime if Self.device.type == "cpu":
            return self._buf[self._coordinates_offset(coords)]
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side item access is not available for GPU"
                        " arrays. Call `.to_host()` first."
                    ),
                    location="AcceleratorNDArray.item(*indices)",
                )
            )

    def itemset(mut self, flat_index: Int, value: Scalar[Self.dtype]) raises:
        comptime if Self.device.type == "cpu":
            var off = self._flat_offset(flat_index)
            self._buf[off] = value
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side itemset is not available for GPU"
                        " arrays. Call `.to_host()`, modify, then"
                        " `.to_device()`."
                    ),
                    location="AcceleratorNDArray.itemset(flat_index)",
                )
            )

    def __getitem__(self) raises -> Scalar[Self.dtype]:
        if self.ndim() != 0 and self.size() != 1:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Use `a[]` only for 0-D or size-1 arrays. Use `item()`"
                        " or regular indexing for other shapes."
                    ),
                    location="AcceleratorNDArray.__getitem__()",
                )
            )
        return self.item(0)

    def __getitem__(self, index: Item) raises -> Scalar[Self.dtype]:
        if index.__len__() != self.ndim():
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices in Item but got {}."
                    ).format(self.ndim(), index.__len__()),
                    location="AcceleratorNDArray.__getitem__(index: Item)",
                )
            )
        var normalized = Item(ndim=self.ndim())
        for i in range(self.ndim()):
            var extent = self._layout.shape_at(i)
            var n = self.normalize(index[i], extent)
            if n < 0 or n >= extent:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for axis {} with size {}."
                        ).format(index[i], i, extent),
                        location="AcceleratorNDArray.__getitem__(index: Item)",
                    )
                )
            normalized[i] = n

        comptime if Self.device.type == "cpu":
            var off = self._coordinates_offset(normalized)
            return self._buf[off]
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side item access is not available for GPU"
                        " arrays. Call `.to_host()` first."
                    ),
                    location="AcceleratorNDArray.__getitem__(index: Item)",
                )
            )

    def __getitem__[
        o: Origin
    ](ref[o] self, idx: Int) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        return self.select(0, idx)

    def __getitem__[
        o: Origin
    ](ref[o] self, var *slices: Slice) raises -> AcceleratorNDArrayView[
        mut=False,
        dtype=Self.dtype,
        device=Self.device,
        origin=ImmOrigin(o),
    ]:
        if self.ndim() == 0:
            raise Error(
                NumojoError(
                    category="index",
                    message="Cannot slice a 0-D array.",
                    location="AcceleratorNDArray.__getitem__(*slices: Slice)",
                )
            )

        var n_slices = len(slices)
        if n_slices > self.ndim():
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Too many slices: got {}, but array ndim is {}."
                    ).format(n_slices, self.ndim()),
                    location="AcceleratorNDArray.__getitem__(*slices: Slice)",
                )
            )

        var new_layout = self._layout.copy()

        for axis in range(self.ndim()):
            var s = slices[axis] if axis < n_slices else Slice(
                0, new_layout.shape_at(axis), 1
            )

            var step = s.step.or_else(1)
            if step == 0:
                raise Error(
                    NumojoError(
                        category="value",
                        message="Slice step cannot be zero.",
                        location=(
                            "AcceleratorNDArray.__getitem__(*slices: Slice)"
                        ),
                    )
                )

            var start, stop, increment = s.indices(new_layout.shape_at(axis))
            var count = 0
            if increment > 0 and start < stop:
                count = 1 + (stop - start - 1) // increment
            elif increment < 0 and start > stop:
                count = 1 + (start - stop - 1) // (-increment)
            new_layout = new_layout.slice(axis, start, count, increment)

        return self._make_view(new_layout^)

    def __setitem__(mut self, index: Item, value: Scalar[Self.dtype]) raises:
        if index.__len__() != self.ndim():
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices in Item but got {}."
                    ).format(self.ndim(), index.__len__()),
                    location=(
                        "AcceleratorNDArray.__setitem__(index: Item, value)"
                    ),
                )
            )
        var coords = List[Int](capacity=self.ndim())
        for i in range(self.ndim()):
            coords.append(index[i])

        comptime if Self.device.type == "cpu":
            var normalized = Item(ndim=self.ndim())
            for i in range(self.ndim()):
                var extent = self._layout.shape_at(i)
                var n = self.normalize(coords[i], extent)
                if n < 0 or n >= extent:
                    raise Error(
                        NumojoError(
                            category="index",
                            message=String(
                                "Index {} out of range for axis {} with"
                                " size {}."
                            ).format(coords[i], i, extent),
                            location=(
                                "AcceleratorNDArray.__setitem__(index: Item,"
                                " value)"
                            ),
                        )
                    )
                normalized[i] = n
            var off = self._coordinates_offset(normalized)
            self._buf[off] = value
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Direct host-side itemset is not available for GPU"
                        " arrays. Call `.to_host()`, modify, then"
                        " `.to_device()`."
                    ),
                    location=(
                        "AcceleratorNDArray.__setitem__(index: Item, value)"
                    ),
                )
            )

    # ===------------------------------------------------------------------=== #
    # Host/Device transfer
    # ===------------------------------------------------------------------=== #

    def to_host(self) raises -> AcceleratorNDArray[Self.dtype, Device.CPU]:
        var out = AcceleratorNDArray[Self.dtype, Device.CPU](shape=self.shape())
        out._layout = self._layout.copy()
        out.flags = self.flags

        comptime if Self.device.type == "cpu":
            var src_ptr = self.unsafe_ptr()
            var dst_ptr = out.unsafe_mut_ptr()
            for i in range(self.size()):
                dst_ptr[unsafe_offset=i] = src_ptr[unsafe_offset=i]
        else:
            self._buf.device_storage.unsafe_value().get_buffer().enqueue_copy_to(
                out.unsafe_mut_ptr()
            )
            self._buf.device_storage.unsafe_value().get_buffer().context().synchronize()

        return out^

    def to_device[
        target: Device
    ](self,) raises -> AcceleratorNDArray[Self.dtype, target]:
        comptime if target.type == "cpu":
            var host = self.to_host()
            var out = AcceleratorNDArray[Self.dtype, target](shape=host.shape())
            out._layout = host._layout.copy()
            out.flags = host.flags
            var src_ptr = host.unsafe_ptr()
            var dst_ptr = out.unsafe_mut_ptr()
            for i in range(host.size()):
                dst_ptr[unsafe_offset=i] = src_ptr[unsafe_offset=i]
            return out^
        else:
            var host = self.to_host()
            var out = AcceleratorNDArray[Self.dtype, target](shape=host.shape())
            out._layout = host._layout.copy()
            out.flags = host.flags
            out._buf.device_storage.unsafe_value().get_buffer().enqueue_copy_from(
                host.unsafe_ptr()
            )
            out._buf.device_storage.unsafe_value().get_buffer().context().synchronize()
            return out^

    def to[
        target: Device
    ](self,) raises -> AcceleratorNDArray[Self.dtype, target]:
        return self.to_device[target]()

    # ===------------------------------------------------------------------=== #
    # Elementwise operations
    # ===------------------------------------------------------------------=== #

    def _binary_op[
        op_code: Int, op_name: StaticString
    ](self, other: Self) raises -> Self:
        """Shared dispatch for elementwise binary operators.

        Both operands must be on the same device and densely contiguous
        (no broadcasting or strided views yet).

        Parameters:
            op_code: One of `kernels.ADD`, `kernels.SUB`, `kernels.MUL`,
                `kernels.DIV`.
            op_name: Human-readable operator name, used in error messages.

        Raises:
            NumojoError: If shapes differ, or either operand is not contiguous.
        """
        if not self._layout.has_same_shape(other._layout):
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Shapes {} and {} do not match for {}."
                    ).format(
                        self.shape().__str__(),
                        other.shape().__str__(),
                        op_name,
                    ),
                    location="AcceleratorNDArray._binary_op",
                )
            )
        # TODO: Relax this constraint later.
        if not self.flags.C_CONTIGUOUS or not other.flags.C_CONTIGUOUS:
            raise Error(
                NumojoError(
                    category="value",
                    message=String(
                        "AcceleratorNDArray {} currently requires both"
                        " operands to be C-contiguous."
                    ).format(op_name),
                    location="AcceleratorNDArray._binary_op",
                )
            )

        var out = Self(shape=self.shape())

        comptime if Self.device.type == "cpu":
            comptime assert Self.device.type == "cpu"
            var dst = out.unsafe_mut_ptr()
            var src1 = self.unsafe_ptr()
            var src2 = other.unsafe_ptr()
            comptime if op_code == kernels.ADD:
                for i in range(self.size()):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] + src2[unsafe_offset=i]
                    )
            elif op_code == kernels.SUB:
                for i in range(self.size()):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] - src2[unsafe_offset=i]
                    )
            elif op_code == kernels.MUL:
                for i in range(self.size()):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] * src2[unsafe_offset=i]
                    )
            else:
                for i in range(self.size()):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] / src2[unsafe_offset=i]
                    )
        else:
            comptime assert Self.device.type == "gpu"
            var context = self.device_context()
            kernels.launch_binary_op[Self.dtype, op_code](
                context,
                out.unsafe_mut_ptr().unsafe_origin_cast[MutAnyOrigin](),
                self.unsafe_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutAnyOrigin](),
                other.unsafe_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutAnyOrigin](),
                self.size(),
            )

        return out^

    def __add__(self, other: Self) raises -> Self:
        """Elementwise addition. See `_binary_op` for constraints."""
        return self._binary_op[kernels.ADD, "addition"](other)

    def __sub__(self, other: Self) raises -> Self:
        """Elementwise subtraction. See `_binary_op` for constraints."""
        return self._binary_op[kernels.SUB, "subtraction"](other)

    def __mul__(self, other: Self) raises -> Self:
        """Elementwise multiplication. See `_binary_op` for constraints."""
        return self._binary_op[kernels.MUL, "multiplication"](other)

    def __truediv__(self, other: Self) raises -> Self:
        """Elementwise division. See `_binary_op` for constraints."""
        return self._binary_op[kernels.DIV, "division"](other)

    def sum(self) raises -> Scalar[Self.dtype]:
        """Sum of all elements in the array. Requires a densely contiguous
        array (no broadcasting or strided views yet).

        Raises:
            NumojoError: If the array is not contiguous.
        """
        if not self.flags.C_CONTIGUOUS:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "AcceleratorNDArray.sum currently requires a"
                        " C-contiguous array."
                    ),
                    location="AcceleratorNDArray.sum",
                )
            )

        comptime if Self.device.type == "cpu":
            comptime assert Self.device.type == "cpu"
            var src = self.unsafe_ptr()
            var result = Scalar[Self.dtype](0)
            for i in range(self.size()):
                result += src[unsafe_offset=i]
            return result
        else:
            comptime assert Self.device.type == "gpu"
            var context = self.device_context()
            return kernels.launch_sum_reduce[Self.dtype](
                context,
                self.unsafe_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutAnyOrigin](),
                self.size(),
            )

    def __neg__(self) raises -> Self:
        """Elementwise negation. Requires a densely contiguous array (no
        broadcasting or strided views yet).

        Raises:
            NumojoError: If the array is not contiguous.
        """
        if not self.flags.C_CONTIGUOUS:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "AcceleratorNDArray.__neg__ currently requires a"
                        " C-contiguous array."
                    ),
                    location="AcceleratorNDArray.__neg__",
                )
            )

        var out = Self(shape=self.shape())

        comptime if Self.device.type == "cpu":
            comptime assert Self.device.type == "cpu"
            var dst = out.unsafe_mut_ptr()
            var src = self.unsafe_ptr()
            for i in range(self.size()):
                dst[unsafe_offset=i] = -src[unsafe_offset=i]
        else:
            comptime assert Self.device.type == "gpu"
            var context = self.device_context()
            kernels.launch_neg[Self.dtype](
                context,
                out.unsafe_mut_ptr().unsafe_origin_cast[MutAnyOrigin](),
                self.unsafe_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutAnyOrigin](),
                self.size(),
            )

        return out^


# ===----------------------------------------------------------------------=== #
# Creation routines
# ===----------------------------------------------------------------------=== #


def empty[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: Shape, order: String = "C") raises -> AcceleratorNDArray[
    dtype, device
]:
    """Create an uninitialized accelerator array on `device`."""
    return AcceleratorNDArray[dtype, device](shape=shape, order=order)


def empty[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: List[Int], order: String = "C") raises -> AcceleratorNDArray[
    dtype, device
]:
    """Create an uninitialized accelerator array on `device`."""
    return empty[dtype, device](Shape(shape), order=order)


def empty[
    dtype: DType = DType.float64, device: Device = Device.CPU
](*shape: Int, order: String = "C") raises -> AcceleratorNDArray[dtype, device]:
    """Create an uninitialized accelerator array on `device`."""
    var dimensions = List[Int](capacity=len(shape))
    for dimension in shape:
        dimensions.append(dimension)
    return empty[dtype, device](Shape(dimensions), order=order)


def full[
    dtype: DType = DType.float64, device: Device = Device.CPU
](
    shape: Shape,
    fill_value: Scalar[dtype],
    order: String = "C",
) raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with `fill_value`."""
    comptime if device.type == "cpu":
        var out = AcceleratorNDArray[dtype, device](shape=shape, order=order)
        for i in range(out.size()):
            out.itemset(i, fill_value)
        return out^
    else:
        var host = full[dtype, Device.CPU](shape, fill_value, order=order)
        return host.to[device]()


def full[
    dtype: DType = DType.float64, device: Device = Device.CPU
](
    shape: List[Int],
    fill_value: Scalar[dtype],
    order: String = "C",
) raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with `fill_value`."""
    return full[dtype, device](Shape(shape), fill_value, order=order)


def zeros[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: Shape, order: String = "C") raises -> AcceleratorNDArray[
    dtype, device
]:
    """Create an accelerator array filled with zeros."""
    return full[dtype, device](shape, Scalar[dtype](0), order=order)


def zeros[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: List[Int], order: String = "C") raises -> AcceleratorNDArray[
    dtype, device
]:
    """Create an accelerator array filled with zeros."""
    return zeros[dtype, device](Shape(shape), order=order)


def zeros[
    dtype: DType = DType.float64, device: Device = Device.CPU
](*shape: Int, order: String = "C") raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with zeros."""
    var dimensions = List[Int](capacity=len(shape))
    for dimension in shape:
        dimensions.append(dimension)
    return zeros[dtype, device](Shape(dimensions), order=order)


def ones[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: Shape, order: String = "C") raises -> AcceleratorNDArray[
    dtype, device
]:
    """Create an accelerator array filled with ones."""
    return full[dtype, device](shape, Scalar[dtype](1), order=order)


def ones[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: List[Int], order: String = "C") raises -> AcceleratorNDArray[
    dtype, device
]:
    """Create an accelerator array filled with ones."""
    return ones[dtype, device](Shape(shape), order=order)


def ones[
    dtype: DType = DType.float64, device: Device = Device.CPU
](*shape: Int, order: String = "C") raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with ones."""
    var dimensions = List[Int](capacity=len(shape))
    for dimension in shape:
        dimensions.append(dimension)
    return ones[dtype, device](Shape(dimensions), order=order)


def empty_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device], order: String = "C"
) raises -> AcceleratorNDArray[dtype, device]:
    """Create an uninitialized accelerator array with `a`'s shape and device."""
    return empty[dtype, device](a.shape(), order=order)


def zeros_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device], order: String = "C"
) raises -> AcceleratorNDArray[dtype, device]:
    """Create a zeros accelerator array with `a`'s shape and device."""
    return zeros[dtype, device](a.shape(), order=order)


def ones_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device], order: String = "C"
) raises -> AcceleratorNDArray[dtype, device]:
    """Create a ones accelerator array with `a`'s shape and device."""
    return ones[dtype, device](a.shape(), order=order)


def full_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device],
    fill_value: Scalar[dtype],
    order: String = "C",
) raises -> AcceleratorNDArray[dtype, device]:
    """Create a filled accelerator array with `a`'s shape and device."""
    return full[dtype, device](a.shape(), fill_value, order=order)


def arange[
    dtype: DType = DType.float64, device: Device = Device.CPU
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype] = Scalar[dtype](1),
) raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array with evenly spaced values."""
    var num = Int((stop - start) / step)
    comptime if device.type == "cpu":
        var out = AcceleratorNDArray[dtype, device](Shape([num]))
        for i in range(num):
            out.itemset(i, start + step * Scalar[dtype](i))
        return out^
    else:
        var host = AcceleratorNDArray[dtype, Device.CPU](Shape([num]))
        for i in range(num):
            host.itemset(i, start + step * Scalar[dtype](i))
        return host.to[device]()


def arange[
    dtype: DType = DType.float64, device: Device = Device.CPU
](stop: Scalar[dtype]) raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array with values from zero to `stop`."""
    return arange[dtype, device](Scalar[dtype](0), stop, Scalar[dtype](1))
