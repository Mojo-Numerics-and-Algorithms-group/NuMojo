# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator NDArray
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""AcceleratorNDArray (numojo.core.accelerator_ndarray)
--------------------------------------------------------
Device-aware NDArray that stores data in `AcceleratorDataContainer`.
"""

from std.memory import UnsafePointer
from max.gpu.host import DeviceContext
from numojo.core.error import NumojoError
from numojo.core.layout.flags import Flags
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.core.indexing.item import Item
from numojo.core.indexing.offset import IndexMethods
from numojo.core.memory.storage import AcceleratorDataContainer
from numojo.core.accelerator.device import Device
import numojo.core.accelerator.kernels as kernels
from numojo.core.ndarray import NDArray
from numojo.core.dtype.default_dtype import _concise_dtype_str
from numojo.core.type_aliases import Shape


struct AcceleratorNDArray[
    dtype: DType = DType.float64, device: Device = Device.CPU
](
    Copyable,
    Movable,
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
    var ndim: Int
    var shape: NDArrayShape
    var size: Int
    var strides: NDArrayStrides
    var offset: Int
    var flags: Flags

    # ===------------------------------------------------------------------=== #
    # Lifecycle
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __init__(out self):
        self._buf = AcceleratorDataContainer[Self.dtype, Self.device]()
        self.ndim = 0
        self.shape = NDArrayShape()
        self.size = 0
        self.strides = NDArrayStrides()
        self.offset = 0
        self.flags = Flags(
            c_contiguous=True,
            f_contiguous=True,
            owndata=False,
            writeable=False,
        )

    @always_inline("nodebug")
    def __init__(out self, shape: NDArrayShape, order: String = "C") raises:
        self.ndim = shape.ndim
        self.shape = shape
        self.size = shape.size()
        self.strides = NDArrayStrides(shape, order=order)
        self.offset = 0
        self._buf = AcceleratorDataContainer[Self.dtype, Self.device](self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

    @always_inline("nodebug")
    def __init__(out self, shape: List[Int], order: String = "C") raises:
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    def __init__(out self, *shape: Int, order: String = "C") raises:
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    def __init__(
        out self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        offset: Int,
        flags: Flags,
    ) raises:
        self.ndim = shape.ndim
        self.shape = shape
        self.size = shape.size()
        self.strides = strides
        self.offset = offset
        self._buf = AcceleratorDataContainer[Self.dtype, Self.device](self.size)
        self.flags = flags

    @always_inline("nodebug")
    def __init__(
        out self,
        var data: AcceleratorDataContainer[Self.dtype, Self.device],
        *,
        is_view: Bool,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        offset: Int,
        size: Int,
    ) raises:
        self._buf = data^
        self.ndim = shape.ndim
        self.shape = shape
        self.size = size
        self.strides = strides
        self.offset = offset
        self.flags = Flags(shape, strides, owndata=not is_view, writeable=True)

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        self.ndim = copy.ndim
        self.shape = copy.shape
        self.size = copy.size
        self.strides = copy.strides
        self.offset = copy.offset
        self.flags = copy.flags
        self._buf = copy._buf.copy()

    @always_inline("nodebug")
    def __init__(out self, *, deinit move: Self):
        self.ndim = move.ndim
        self.shape = move.shape
        self.size = move.size
        self.strides = move.strides
        self.offset = move.offset
        self.flags = move.flags
        self._buf = move._buf^

    @always_inline("nodebug")
    def view(self) raises -> Self:
        """Create a metadata-only view sharing the same storage."""
        return self._make_view(
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            size=self.size,
        )

    @always_inline("nodebug")
    def _make_view(
        self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        offset: Int,
        size: Int,
    ) raises -> Self:
        var shared = self._buf.share()
        return Self(
            shared^,
            is_view=True,
            shape=shape,
            strides=strides,
            offset=offset,
            size=size,
        )

    # ===------------------------------------------------------------------=== #
    # Representation
    # ===------------------------------------------------------------------=== #

    def _as_cpu_ndarray_for_display(self) raises -> NDArray[Self.dtype]:
        var out = NDArray[Self.dtype](self.shape)
        for i in range(self.size):
            out.itemset(i, self.item(i))
        return out^

    def __str__(self) -> String:
        var res: String
        try:
            if self.ndim == 0:
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
                    + String(self.ndim)
                    + "D-array  Shape: "
                    + self.shape.__str__()
                    + "  Strides: "
                    + self.strides.__str__()
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
        if self.ndim == 0:
            return 1
        return Int(self.shape.unsafe_load(0))

    # ===------------------------------------------------------------------=== #
    # Device helpers
    # ===------------------------------------------------------------------=== #

    @parameter
    def is_cpu(self) -> Bool:
        return Self.device.type == "cpu"

    @parameter
    def is_gpu(self) -> Bool:
        return Self.device.type == "gpu"

    def unsafe_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin] where (
        Self.device.type == "cpu"
    ):
        return (
            self._buf.host_storage.unsafe_value()
            .ptr.unsafe_offset(self.offset)
            .as_unsafe_any_origin()
        )

    def unsafe_device_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin] where (
        Self.device.type == "gpu"
    ):
        """Return the raw device pointer to the buffer's data.

        Returns:
            An `UnsafePointer` to the first element of the underlying
            device buffer (not the logical view start).
        """
        return self._buf.device_storage.unsafe_value().unsafe_ptr()

    def device_context(
        self,
    ) -> DeviceContext where Self.device.type == "gpu":
        """Return the `DeviceContext` backing this array's GPU storage."""
        return self._buf.device_storage.unsafe_value().handle.device_context()

    def num_elements(self) -> Int:
        return self.size

    # ===------------------------------------------------------------------=== #
    # Indexing and scalar access
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def normalize(self, index: Int, dim: Int) -> Int:
        return index if index >= 0 else index + dim

    @always_inline("nodebug")
    def _flat_offset(self, flat_index: Int) raises -> Int:
        var idx = self.normalize(flat_index, self.size)
        if idx < 0 or idx >= self.size:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Flat index {} out of range for size {}."
                    ).format(flat_index, self.size),
                    location="AcceleratorNDArray._flat_offset",
                )
            )
        if self.flags.C_CONTIGUOUS or self.ndim <= 1:
            return self.offset + idx

        var rem = idx
        var coords = Item(ndim=self.ndim)
        for d in range(self.ndim - 1, -1, -1):
            coords[d] = rem % self.shape[d]
            rem //= self.shape[d]
        return self.offset + IndexMethods.get_1d_index(coords, self.strides)

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
        if len(indices) != self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String("Expected {} indices but got {}.").format(
                        self.ndim, len(indices)
                    ),
                    location="AcceleratorNDArray.item(*indices)",
                )
            )

        var coords = Item(ndim=self.ndim)
        for i in range(self.ndim):
            var n = self.normalize(indices[i], self.shape[i])
            if n < 0 or n >= self.shape[i]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for axis {} with size {}."
                        ).format(indices[i], i, self.shape[i]),
                        location="AcceleratorNDArray.item(*indices)",
                    )
                )
            coords[i] = n

        comptime if Self.device.type == "cpu":
            return self._buf[
                self.offset + IndexMethods.get_1d_index(coords, self.strides)
            ]
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
        if self.ndim != 0 and self.size != 1:
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
        if index.__len__() != self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices in Item but got {}."
                    ).format(self.ndim, index.__len__()),
                    location="AcceleratorNDArray.__getitem__(index: Item)",
                )
            )
        var normalized = Item(ndim=self.ndim)
        for i in range(self.ndim):
            var n = self.normalize(index[i], self.shape[i])
            if n < 0 or n >= self.shape[i]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for axis {} with size {}."
                        ).format(index[i], i, self.shape[i]),
                        location="AcceleratorNDArray.__getitem__(index: Item)",
                    )
                )
            normalized[i] = n

        comptime if Self.device.type == "cpu":
            var off = self.offset + IndexMethods.get_1d_index(
                normalized, self.strides
            )
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

    def __getitem__(self, idx: Int) raises -> Self:
        if self.ndim == 0:
            raise Error(
                NumojoError(
                    category="index",
                    message="Cannot index a 0-D array with an integer index.",
                    location="AcceleratorNDArray.__getitem__(idx: Int)",
                )
            )

        var norm = self.normalize(idx, self.shape[0])
        if norm < 0 or norm >= self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} out of range for axis 0 with size {}."
                    ).format(idx, self.shape[0]),
                    location="AcceleratorNDArray.__getitem__(idx: Int)",
                )
            )

        var new_offset = self.offset + norm * self.strides[0]
        if self.ndim == 1:
            return self._make_view(
                shape=NDArrayShape(),
                strides=NDArrayStrides(),
                offset=new_offset,
                size=1,
            )

        var new_shape = self.shape.pop(0)
        var new_strides = self.strides.pop(0)
        var new_size = self.size // self.shape[0]
        return self._make_view(
            shape=new_shape,
            strides=new_strides,
            offset=new_offset,
            size=new_size,
        )

    def __getitem__(self, var *slices: Slice) raises -> Self:
        if self.ndim == 0:
            raise Error(
                NumojoError(
                    category="index",
                    message="Cannot slice a 0-D array.",
                    location="AcceleratorNDArray.__getitem__(*slices: Slice)",
                )
            )

        var n_slices = len(slices)
        if n_slices > self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Too many slices: got {}, but array ndim is {}."
                    ).format(n_slices, self.ndim),
                    location="AcceleratorNDArray.__getitem__(*slices: Slice)",
                )
            )

        var new_shape = List[Int](capacity=self.ndim)
        var new_strides = List[Int](capacity=self.ndim)
        var new_offset = self.offset
        var new_size = 1

        for axis in range(self.ndim):
            var s = slices[axis] if axis < n_slices else Slice(
                0, self.shape[axis], 1
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

            var dim = self.shape[axis]
            var start: Int
            var end: Int

            if step > 0:
                start = s.start.or_else(0)
                end = s.end.or_else(dim)
            else:
                start = s.start.or_else(dim - 1)
                end = s.end.or_else(-1)

            if start < 0:
                start += dim
            if end < 0:
                end += dim

            if step > 0:
                if start < 0:
                    start = 0
                if start > dim:
                    start = dim
                if end < 0:
                    end = 0
                if end > dim:
                    end = dim
            else:
                if start < -1:
                    start = -1
                if start >= dim:
                    start = dim - 1
                if end < -1:
                    end = -1
                if end >= dim:
                    end = dim - 1

            var slice_len: Int
            if step > 0:
                slice_len = max((end - start + step - 1) // step, 0)
            else:
                slice_len = max((start - end + (-step) - 1) // (-step), 0)

            if slice_len == 0:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "Empty slices are not supported yet in"
                            " AcceleratorNDArray basic slicing."
                        ),
                        location=(
                            "AcceleratorNDArray.__getitem__(*slices: Slice)"
                        ),
                    )
                )

            new_offset += start * self.strides[axis]
            new_shape.append(slice_len)
            new_strides.append(self.strides[axis] * step)
            new_size *= slice_len

        return self._make_view(
            shape=NDArrayShape(new_shape),
            strides=NDArrayStrides(strides=new_strides),
            offset=new_offset,
            size=new_size,
        )

    def __setitem__(mut self, index: Item, value: Scalar[Self.dtype]) raises:
        if index.__len__() != self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices in Item but got {}."
                    ).format(self.ndim, index.__len__()),
                    location=(
                        "AcceleratorNDArray.__setitem__(index: Item, value)"
                    ),
                )
            )
        var coords = List[Int](capacity=self.ndim)
        for i in range(self.ndim):
            coords.append(index[i])

        comptime if Self.device.type == "cpu":
            var normalized = Item(ndim=self.ndim)
            for i in range(self.ndim):
                var n = self.normalize(coords[i], self.shape[i])
                if n < 0 or n >= self.shape[i]:
                    raise Error(
                        NumojoError(
                            category="index",
                            message=String(
                                "Index {} out of range for axis {} with"
                                " size {}."
                            ).format(coords[i], i, self.shape[i]),
                            location=(
                                "AcceleratorNDArray.__setitem__(index: Item,"
                                " value)"
                            ),
                        )
                    )
                normalized[i] = n
            var off = self.offset + IndexMethods.get_1d_index(
                normalized, self.strides
            )
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

    def deep_copy(self) raises -> Self:
        var out = Self(shape=self.shape)
        out.strides = self.strides
        out.offset = self.offset
        out.flags = self.flags

        comptime if Self.device.type == "cpu":
            var src_ptr = self._buf.host_storage.unsafe_value().ptr
            var dst_ptr = out._buf.host_storage.unsafe_value().ptr
            for i in range(self.size):
                dst_ptr[unsafe_offset=i] = src_ptr[unsafe_offset=i]
        else:
            self._buf.device_storage.unsafe_value().get_buffer().enqueue_copy_to(
                out._buf.device_storage.unsafe_value().get_buffer()
            )
            self._buf.device_storage.unsafe_value().get_buffer().context().synchronize()
        return out^

    def to_host(self) raises -> AcceleratorNDArray[Self.dtype, Device.CPU]:
        var out = AcceleratorNDArray[Self.dtype, Device.CPU](shape=self.shape)
        out.strides = self.strides
        out.offset = self.offset
        out.flags = self.flags

        comptime if Self.device.type == "cpu":
            var src_ptr = self._buf.host_storage.unsafe_value().ptr
            var dst_ptr = out._buf.host_storage.unsafe_value().ptr
            for i in range(self.size):
                dst_ptr[unsafe_offset=i] = src_ptr[unsafe_offset=i]
        else:
            self._buf.device_storage.unsafe_value().get_buffer().enqueue_copy_to(
                out._buf.host_storage.unsafe_value().ptr
            )
            self._buf.device_storage.unsafe_value().get_buffer().context().synchronize()

        return out^

    def to_device[
        target: Device
    ](self,) raises -> AcceleratorNDArray[Self.dtype, target]:
        comptime if target.type == "cpu":
            var host = self.to_host()
            var out = AcceleratorNDArray[Self.dtype, target](shape=host.shape)
            out.strides = host.strides
            out.offset = host.offset
            out.flags = host.flags
            var src_ptr = host._buf.host_storage.unsafe_value().ptr
            var dst_ptr = out._buf.host_storage.unsafe_value().ptr
            for i in range(host.size):
                dst_ptr[unsafe_offset=i] = src_ptr[unsafe_offset=i]
            return out^
        else:
            var host = self.to_host()
            var out = AcceleratorNDArray[Self.dtype, target](shape=host.shape)
            out.strides = host.strides
            out.offset = host.offset
            out.flags = host.flags
            out._buf.device_storage.unsafe_value().get_buffer().enqueue_copy_from(
                host._buf.host_storage.unsafe_value().ptr
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
            Error: If shapes differ, or either operand is not contiguous.
        """
        if self.shape != other.shape:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Shapes {} and {} do not match for {}."
                    ).format(self.shape, other.shape, op_name),
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

        var out = Self(shape=self.shape)

        comptime if Self.device.type == "cpu":
            comptime assert Self.device.type == "cpu"
            var dst = out.unsafe_ptr()
            var src1 = self.unsafe_ptr()
            var src2 = other.unsafe_ptr()
            comptime if op_code == kernels.ADD:
                for i in range(self.size):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] + src2[unsafe_offset=i]
                    )
            elif op_code == kernels.SUB:
                for i in range(self.size):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] - src2[unsafe_offset=i]
                    )
            elif op_code == kernels.MUL:
                for i in range(self.size):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] * src2[unsafe_offset=i]
                    )
            else:
                for i in range(self.size):
                    dst[unsafe_offset=i] = (
                        src1[unsafe_offset=i] / src2[unsafe_offset=i]
                    )
        else:
            comptime assert Self.device.type == "gpu"
            var context = self.device_context()
            kernels.launch_binary_op[Self.dtype, op_code](
                context,
                out.unsafe_device_ptr(),
                self.unsafe_device_ptr(),
                other.unsafe_device_ptr(),
                self.size,
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
            Error: If the array is not contiguous.
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
            for i in range(self.size):
                result += src[unsafe_offset=i]
            return result
        else:
            comptime assert Self.device.type == "gpu"
            var context = self.device_context()
            return kernels.launch_sum_reduce[Self.dtype](
                context, self.unsafe_device_ptr(), self.size
            )

    def __neg__(self) raises -> Self:
        """Elementwise negation. Requires a densely contiguous array (no
        broadcasting or strided views yet).

        Raises:
            Error: If the array is not contiguous.
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

        var out = Self(shape=self.shape)

        comptime if Self.device.type == "cpu":
            comptime assert Self.device.type == "cpu"
            var dst = out.unsafe_ptr()
            var src = self.unsafe_ptr()
            for i in range(self.size):
                dst[unsafe_offset=i] = -src[unsafe_offset=i]
        else:
            comptime assert Self.device.type == "gpu"
            var context = self.device_context()
            kernels.launch_neg[Self.dtype](
                context,
                out.unsafe_device_ptr(),
                self.unsafe_device_ptr(),
                self.size,
            )

        return out^


# ===----------------------------------------------------------------------=== #
# Creation routines
# ===----------------------------------------------------------------------=== #


def empty[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: NDArrayShape, order: String = "C") raises -> AcceleratorNDArray[
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
    return empty[dtype, device](NDArrayShape(shape), order=order)


def empty[
    dtype: DType = DType.float64, device: Device = Device.CPU
](*shape: Int, order: String = "C") raises -> AcceleratorNDArray[dtype, device]:
    """Create an uninitialized accelerator array on `device`."""
    return empty[dtype, device](NDArrayShape(shape), order=order)


def full[
    dtype: DType = DType.float64, device: Device = Device.CPU
](
    shape: NDArrayShape,
    fill_value: Scalar[dtype],
    order: String = "C",
) raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with `fill_value`."""
    comptime if device.type == "cpu":
        var out = AcceleratorNDArray[dtype, device](shape=shape, order=order)
        for i in range(out.size):
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
    return full[dtype, device](NDArrayShape(shape), fill_value, order=order)


def zeros[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: NDArrayShape, order: String = "C") raises -> AcceleratorNDArray[
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
    return zeros[dtype, device](NDArrayShape(shape), order=order)


def zeros[
    dtype: DType = DType.float64, device: Device = Device.CPU
](*shape: Int, order: String = "C") raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with zeros."""
    return zeros[dtype, device](NDArrayShape(shape), order=order)


def ones[
    dtype: DType = DType.float64, device: Device = Device.CPU
](shape: NDArrayShape, order: String = "C") raises -> AcceleratorNDArray[
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
    return ones[dtype, device](NDArrayShape(shape), order=order)


def ones[
    dtype: DType = DType.float64, device: Device = Device.CPU
](*shape: Int, order: String = "C") raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array filled with ones."""
    return ones[dtype, device](NDArrayShape(shape), order=order)


def empty_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device], order: String = "C"
) raises -> AcceleratorNDArray[dtype, device]:
    """Create an uninitialized accelerator array with `a`'s shape and device."""
    return empty[dtype, device](a.shape, order=order)


def zeros_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device], order: String = "C"
) raises -> AcceleratorNDArray[dtype, device]:
    """Create a zeros accelerator array with `a`'s shape and device."""
    return zeros[dtype, device](a.shape, order=order)


def ones_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device], order: String = "C"
) raises -> AcceleratorNDArray[dtype, device]:
    """Create a ones accelerator array with `a`'s shape and device."""
    return ones[dtype, device](a.shape, order=order)


def full_like[
    dtype: DType, device: Device
](
    a: AcceleratorNDArray[dtype, device],
    fill_value: Scalar[dtype],
    order: String = "C",
) raises -> AcceleratorNDArray[dtype, device]:
    """Create a filled accelerator array with `a`'s shape and device."""
    return full[dtype, device](a.shape, fill_value, order=order)


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
        var out = AcceleratorNDArray[dtype, device](Shape(num))
        for i in range(num):
            out.itemset(i, start + step * Scalar[dtype](i))
        return out^
    else:
        var host = AcceleratorNDArray[dtype, Device.CPU](Shape(num))
        for i in range(num):
            host.itemset(i, start + step * Scalar[dtype](i))
        return host.to[device]()


def arange[
    dtype: DType = DType.float64, device: Device = Device.CPU
](stop: Scalar[dtype]) raises -> AcceleratorNDArray[dtype, device]:
    """Create an accelerator array with values from zero to `stop`."""
    return arange[dtype, device](Scalar[dtype](0), stop, Scalar[dtype](1))
