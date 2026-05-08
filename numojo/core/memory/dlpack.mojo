# ===----------------------------------------------------------------------=== #
# NuMojo: DLPack Interop Module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""DLPack (numojo.core.memory.dlpack)

This module implements the DLPack protocol for zero-copy tensor exchange
between NuMojo and other array libraries (NumPy, PyTorch, JAX, etc.).

DLPack is an open standard for in-memory tensor structures that enables
zero-copy data sharing between different frameworks.

References:
    - DLPack Specification: https://dmlc.github.io/dlpack/latest/

Example:
    ```mojo
    from numojo.prelude import *
    from numojo.core.memory.dlpack import from_dlpack
    from python import Python

    def main() raises:
        # Create a NuMojo array
        var arr = nm.linspace[f32](0, 5, 6)

        # Import NumPy array back to NuMojo via DLPack
        var np = Python.import_module("numpy")
        var numpy_data = np.linspace(0, 5, 6, dtype=np.float32)
        var mojo_arr = from_dlpack[f32](numpy_data)
        print(mojo_arr)

        # Import PyTorch tensor to NuMojo via DLPack
        var torch = Python.import_module("torch")
        var torch_tensor = torch.rand(Python.tuple(4, 4), dtype=torch.float64)
        var mojo_tensor = from_dlpack[f64](torch_tensor)
        print(mojo_tensor)
    ```
"""

from std.memory import UnsafePointer
from std.sys.info import size_of
from std.python import PythonObject, Python

from numojo.core.ndarray import NDArray
from numojo.core.memory.data_container import DataContainer
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.layout.ndstrides import NDArrayStrides


# ===-------------------------------------------------------------------===#
# DLPack Core Structures
# ===-------------------------------------------------------------------===#


# TODO: Some of these correspond to older DLPack versions. Need to upgrade this to v1.0
struct DLPackVersion(ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Represents a DLPack version structure for compatibility checking.

    This structure stores major and minor version numbers to ensure compatibility
    between different implementations of the DLPack protocol. The current
    implementation targets DLPack version 0.8.

    Attributes:
        major: Major version number.
        minor: Minor version number.

    Constants:
        CURRENT_MAJOR: Current major version (0).
        CURRENT_MINOR: Current minor version (8).
        LATEST: Latest version instance.
    """

    var major: UInt32
    var minor: UInt32

    comptime CURRENT_MAJOR: UInt32 = 0
    comptime CURRENT_MINOR: UInt32 = 8
    comptime LATEST = DLPackVersion(Self.CURRENT_MAJOR, Self.CURRENT_MINOR)

    def __init__(out self, major: UInt32, minor: UInt32):
        self.major = major
        self.minor = minor


struct DLDevice(ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Represents a device context for tensor data.

    Describes where the tensor data is physically located (CPU, GPU, etc.)
    and which specific device instance to use if there are multiple devices
    of the same type.

    Attributes:
        device_type: Device type code (CPU=1, CUDA=2, OPENCL=4, etc.).
        device_id: Device ID for multiple devices of the same type (usually 0).

    Constants:
        CPU: CPU device type code (1).
        CUDA: CUDA GPU device type code (2).
        OPENCL: OpenCL device type code (4).
        VULKAN: Vulkan device type code (7).
        METAL: Metal device type code (8).
        VPI: VPI device type code (9).
        ROCM: ROCm device type code (10).
    """

    # TODO: verify all the values apart from cpu, cuda.
    comptime CPU = 1
    comptime CUDA = 2
    comptime OPENCL = 4
    comptime VULKAN = 7
    comptime METAL = 8
    comptime VPI = 9
    comptime ROCM = 10

    var device_type: Int32
    """Device type code."""
    var device_id: Int32
    """Device ID (for multiple devices of same type)."""

    def __init__(out self, device_type: Int32 = Self.CPU, device_id: Int32 = 0):
        self.device_type = device_type
        self.device_id = device_id


struct DLDataType(ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Represents a data type descriptor for tensor elements.

    Describes the element type using a type code (int/float/complex/bool),
    bit width (8, 16, 32, 64, etc.), and number of lanes for vector types.

    Attributes:
        code: Type code (INT=0, UINT=1, FLOAT=2, BFLOAT=4, COMPLEX=5, BOOL=6).
        bits: Number of bits per element (8, 16, 32, 64, etc.).
        lanes: Number of lanes (1 for scalar types, >1 for vector types).

    Constants:
        INT: Signed integer type code (0).
        UINT: Unsigned integer type code (1).
        FLOAT: Floating-point type code (2).
        BFLOAT: Brain floating-point type code (4).
        COMPLEX: Complex number type code (5).
        BOOL: Boolean type code (6).
    """

    comptime INT = 0
    comptime UINT = 1
    comptime FLOAT = 2
    comptime BFLOAT = 4
    comptime COMPLEX = 5
    comptime BOOL = 6

    var code: UInt8
    """Type code (INT, UINT, FLOAT, etc.)."""
    var bits: UInt8
    """Number of bits per element."""
    var lanes: UInt16
    """Number of lanes (1 for scalar, >1 for vector types)."""

    def __init__(out self, code: UInt8, bits: UInt8, lanes: UInt16 = 1):
        self.code = code
        self.bits = bits
        self.lanes = lanes

    @staticmethod
    def from_dtype[dtype: DType]() -> Self:
        """Converts a Mojo DType to a DLDataType descriptor.

        This static method maps Mojo's native data types to the DLPack type
        system, determining the appropriate type code (INT, UINT, FLOAT) and bit
        width based on the input DType.

        Parameters:
            dtype: Mojo data type to convert.

        Returns:
            Corresponding DLDataType descriptor with appropriate type code and
            bit width.
        """
        var code: UInt8
        var bits: UInt8 = UInt8(size_of[dtype]() * 8)

        if dtype.is_integral():
            if dtype.is_signed():
                code = Self.INT
            else:
                code = Self.UINT
        elif dtype.is_floating_point():
            code = Self.FLOAT
        else:
            code = Self.FLOAT

        return Self(code, bits, 1)

    def to_dtype(self) raises -> DType:
        """Converts a DLDataType descriptor to a Mojo DType.

        This method maps DLPack type descriptors back to Mojo's native data types,
        supporting common floating-point (float16/32/64) and integer types
        (int8/16/32/64, uint8/16/32/64).

        Returns:
            Corresponding Mojo DType.

        Raises:
            Error: If the type code is not supported.
            Error: If the bit width is not supported for the given type code.
            Error: If vector types (lanes > 1) are encountered (not yet
                supported).
        """
        if self.lanes != 1:
            raise Error("DLDataType: vector types not supported")

        # TODO: Implement other Mojo datatypes if possible.
        if self.code == Self.FLOAT:
            if self.bits == 32:
                return DType.float32
            elif self.bits == 64:
                return DType.float64
            elif self.bits == 16:
                return DType.float16
            else:
                raise Error(
                    "DLDataType: unsupported float bit width: "
                    + String(self.bits)
                )
        elif self.code == Self.INT:
            if self.bits == 8:
                return DType.int8
            elif self.bits == 16:
                return DType.int16
            elif self.bits == 32:
                return DType.int32
            elif self.bits == 64:
                return DType.int64
            else:
                raise Error(
                    "DLDataType: unsupported int bit width: "
                    + String(self.bits)
                )
        elif self.code == Self.UINT:
            if self.bits == 8:
                return DType.uint8
            elif self.bits == 16:
                return DType.uint16
            elif self.bits == 32:
                return DType.uint32
            elif self.bits == 64:
                return DType.uint64
            else:
                raise Error(
                    "DLDataType: unsupported uint bit width: "
                    + String(self.bits)
                )
        else:
            raise Error(
                "DLDataType: unsupported type code: " + String(self.code)
            )


struct DLTensor(ImplicitlyCopyable, Movable):
    """Represents the core tensor structure containing data pointer and metadata.

    This is the fundamental structure that describes a tensor's memory layout,
    shape, strides, and data type without managing its lifetime. It provides
    a view into tensor data without ownership semantics.

    Attributes:
        data: Opaque pointer to the tensor data.
        device: Device where the data resides.
        ndim: Number of dimensions.
        dtype: Element data type descriptor.
        shape: Pointer to shape array (size = ndim).
        strides: Pointer to strides array in elements (size = ndim).
        byte_offset: Byte offset from data pointer to first element.
    """

    var data: UnsafePointer[NoneType, MutAnyOrigin]
    """Opaque pointer to the tensor data."""
    var device: DLDevice
    """Device where the data resides."""
    var ndim: Int32
    """Number of dimensions."""
    var dtype: DLDataType
    """Element data type."""
    var shape: UnsafePointer[Int64, MutAnyOrigin]
    """Shape array."""
    var strides: UnsafePointer[Int64, MutAnyOrigin]
    """Strides in elements."""
    var byte_offset: UInt64
    """Byte offset from data pointer to first element."""

    def __init__(
        out self,
        data: UnsafePointer[NoneType, MutAnyOrigin],
        device: DLDevice,
        ndim: Int32,
        dtype: DLDataType,
        shape: UnsafePointer[Int64, MutAnyOrigin],
        strides: UnsafePointer[Int64, MutAnyOrigin],
        byte_offset: UInt64 = 0,
    ):
        self.data = data
        self.device = device
        self.ndim = ndim
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.byte_offset = byte_offset


comptime DLManagedTensorDeleter = def(
    UnsafePointer[DLManagedTensor, MutAnyOrigin]
) capturing -> None


# TODO: This is the old API, current API requires using DLManagedTensorVersioned which has a version field for future compatibility.
# Need to update this to match the latest DLPack spec.
struct DLManagedTensor(ImplicitlyCopyable, Movable):
    """Represents a managed tensor structure that includes a deleter callback
    for lifetime management.

    This structure wraps a DLTensor with a deleter callback and optional
    context pointer for resource cleanup. The deleter is called by the
    consumer when they're done with the data, ensuring proper memory management
    in cross-framework data sharing scenarios.

    Attributes:
        dl_tensor: The underlying tensor structure.
        manager_ctx: Context pointer for the deleter (stores metadata, refcount,
            etc.).
        deleter: Cleanup function called when consumer finishes using the tensor.

    Note:
        This implements the older DLPack API. The current specification uses
        DLManagedTensorVersioned with a version field for forward compatibility.
    """

    var dl_tensor: DLTensor
    """The underlying tensor."""
    var manager_ctx: UnsafePointer[NoneType, MutAnyOrigin]
    """Context pointer for the deleter (stores metadata, refcount, etc.)."""
    var deleter: DLManagedTensorDeleter
    """Cleanup function called when consumer is done."""

    def __init__(
        out self,
        dl_tensor: DLTensor,
        manager_ctx: UnsafePointer[NoneType, MutAnyOrigin],
        deleter: DLManagedTensorDeleter,
    ):
        self.dl_tensor = dl_tensor.copy()
        self.manager_ctx = manager_ctx
        self.deleter = deleter


# ===-------------------------------------------------------------------===#
# Metadata Management
# ===-------------------------------------------------------------------===#


struct DLPackMetadata[dtype: DType](ImplicitlyCopyable, Movable):
    """Represents a metadata container for DLPack tensor lifetime management.

    This structure stores all the metadata needed to manage the lifetime of a
    DLPack tensor, including shape, strides, and the underlying data container.
    It ensures proper cleanup of allocated resources when the tensor is no
    longer needed.

    Parameters:
        dtype: Data type of the tensor elements.

    Attributes:
        shape: Pointer to shape array.
        strides: Pointer to strides array.
        ndim: Number of dimensions.
        data_container: Container managing the actual tensor data.
    """

    var shape: UnsafePointer[Int64, MutAnyOrigin]
    var strides: UnsafePointer[Int64, MutAnyOrigin]
    var ndim: Int
    var data_container: DataContainer[Self.dtype]

    def __init__(
        out self,
        shape: UnsafePointer[Int64, MutAnyOrigin],
        strides: UnsafePointer[Int64, MutAnyOrigin],
        ndim: Int,
        var data_container: DataContainer[Self.dtype],
    ):
        self.shape = shape
        self.strides = strides
        self.ndim = ndim
        self.data_container = data_container^

    def __copyinit__(out self, copy: Self):
        self.shape = copy.shape
        self.strides = copy.strides
        self.ndim = copy.ndim
        self.data_container = copy.data_container.copy()

    def __del__(deinit self):
        if self.shape:
            self.shape.free()
        if self.strides:
            self.strides.free()
        # TODO: note sure if we should free it explicitly, gotta check this.
        _ = self.data_container^


# ===-------------------------------------------------------------------===#
# Deleter Callbacks
# ===-------------------------------------------------------------------===#


def _dlpack_deleter_impl[
    dtype: DType
](managed_tensor_ptr: UnsafePointer[DLManagedTensor, MutAnyOrigin]) capturing -> None:
    """Type-specific deleter callback for DLManagedTensor."""
    if not managed_tensor_ptr:
        return

    var ctx = managed_tensor_ptr[].manager_ctx
    if ctx:
        var metadata_ptr = ctx.bitcast[DLPackMetadata[dtype]]()
        metadata_ptr.destroy_pointee()
        metadata_ptr.free()

    managed_tensor_ptr.free()


# ===-------------------------------------------------------------------===#
# Export Functions
# ===-------------------------------------------------------------------===#


def to_dlpack[
    dtype: DType
](arr: NDArray[dtype]) raises -> UnsafePointer[DLManagedTensor, MutAnyOrigin]:
    """Exports a NuMojo NDArray to a DLPack managed tensor for zero-copy sharing.

    This function converts a NuMojo NDArray into a DLPack-compatible managed
    tensor that can be consumed by other array libraries (NumPy, PyTorch, JAX,
    etc.) without copying the underlying data. The function allocates shape and
    strides arrays, creates metadata for lifetime management, and returns a
    pointer to a DLManagedTensor.

    Parameters:
        dtype: Data type of the array elements.

    Args:
        arr: The NDArray to export.

    Returns:
        Pointer to a DLManagedTensor that can be consumed by other libraries.

    Raises:
        Error: If enabling views on the data container fails.

    Notes:
        - The consumer is responsible for calling the deleter when done.
        - Do not modify the original array while the DLPack tensor is in use.
        - The returned tensor shares memory with the original array.
    """
    var buf = arr._buf.copy()

    var shape_ptr = alloc[Int64](arr.ndim)
    for i in range(arr.ndim):
        shape_ptr[i] = Int64(arr.shape[i])

    var strides_ptr = alloc[Int64](arr.ndim)
    for i in range(arr.ndim):
        strides_ptr[i] = Int64(arr.strides[i])

    var dl_dtype = DLDataType.from_dtype[dtype]()
    var device = DLDevice(DLDevice.CPU, 0)
    var data_ptr = buf.get_ptr().bitcast[NoneType]()

    var dl_tensor = DLTensor(
        data=data_ptr,
        device=device,
        ndim=Int32(arr.ndim),
        dtype=dl_dtype,
        shape=shape_ptr,
        strides=strides_ptr,
        byte_offset=0,
    )

    var metadata = DLPackMetadata[dtype](
        shape_ptr,
        strides_ptr,
        arr.ndim,
        buf^,
    )

    var ctx = alloc[DLPackMetadata[dtype]](1)
    ctx.init_pointee_move(metadata^)

    var managed = alloc[DLManagedTensor](1)
    managed.init_pointee_move(
        DLManagedTensor(
            dl_tensor,
            ctx.bitcast[NoneType](),
            _dlpack_deleter_impl[dtype],
        )
    )

    return managed


def _extract_dlpack_pointer(
    capsule: PythonObject,
) raises -> UnsafePointer[DLManagedTensor, MutAnyOrigin]:
    """Extracts the DLManagedTensor pointer from a PyCapsule.

    This function uses the Python C API to extract the raw pointer from a
    PyCapsule object. It dynamically loads libpython, calls
    PyCapsule_GetPointer, and validates the result before returning the pointer.

    Args:
        capsule: A PythonObject that is expected to be a PyCapsule containing a
            pointer to a DLManagedTensor. This is typically obtained from the
            __dlpack__() method of a DLPack-compatible tensor.

    Returns:
        Pointer to the DLManagedTensor inside the capsule.

    Raises:
        Error: If PyCapsule_GetPointer returns NULL (capsule may be invalid or
        already consumed).
    """
    ctypes = Python.import_module("ctypes")

    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object,
        ctypes.c_char_p,
    ]
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    result_obj = ctypes.pythonapi.PyCapsule_GetPointer(
        capsule, _get_c_char_p_from_string["dltensor"]()
    )

    var p = result_obj.unsafe_get_as_pointer[DType.uint8]()

    if not p:
        raise Error(
            "_extract_dlpack_pointer: PyCapsule_GetPointer returned NULL"
            " - capsule may be invalid or already consumed"
        )

    return p.bitcast[DLManagedTensor]()


def _get_c_char_p_from_string[s: StringLiteral]() raises -> PythonObject:
    ctypes = Python.import_module("ctypes")

    return ctypes.cast(
        Int(s.as_c_string_slice().unsafe_ptr().bitcast[NoneType]()),
        ctypes.c_char_p,
    )


# ===-------------------------------------------------------------------===#
# Import Functions
# ===-------------------------------------------------------------------===#


def from_dlpack[dtype: DType](capsule: PythonObject) raises -> NDArray[dtype]:
    """Imports a tensor from any DLPack-compatible library into a NuMojo NDArray
    using zero-copy.
    This function accepts a Python object that implements the DLPack protocol
    (i.e., has a __dlpack__() method), such as NumPy, PyTorch, JAX, or CuPy
    tensors. It extracts the underlying memory and metadata through the
    PyCapsule interface, validates device and data type compatibility, and
    returns a NuMojo NDArray that shares memory with the original tensor.

    Parameters:
        dtype: The expected data type of the array elements.

    Args:
        capsule: A PythonObject representing a DLPack-compatible tensor. The
            object must implement the __dlpack__() method, which returns a
            PyCapsule containing a pointer to a DLManagedTensor.

    Returns:
        A new NuMojo NDArray that shares memory with the input tensor.

    Raises:
        Error: If the received DLManagedTensor pointer is null.
        Error: If the tensor is not on CPU (only CPU tensors are currently
            supported).
        Error: If the data type does not match the expected dtype parameter.

    Notes:
        - The returned NDArray shares memory with the source tensor. Changes to
            one will be reflected in the other.
        - Only CPU tensors are currently supported.
        - If strides are not provided in the DLPack tensor, C-contiguous strides
            are assumed.
    """
    # This assumes that the input Python object has a __dlpack__ method that
    # returns a PyCapsule containing a pointer to a DLManagedTensor.
    # Perhaps we should add a check to verify that in future.
    var actual_capsule: PythonObject = capsule.__dlpack__()
    var managed_tensor_ptr = _extract_dlpack_pointer(actual_capsule)

    if not managed_tensor_ptr:
        raise Error("from_dlpack: received null DLManagedTensor pointer")

    var dl_tensor = managed_tensor_ptr[].dl_tensor

    if dl_tensor.device.device_type != DLDevice.CPU:
        raise Error(
            "from_dlpack: only CPU tensors are currently supported, got"
            " device type "
            + String(Int(dl_tensor.device.device_type))
        )

    var received_dtype = dl_tensor.dtype.to_dtype()
    if received_dtype != dtype:
        raise Error(
            "from_dlpack: dtype mismatch - expected "
            + String(dtype)
            + " but got "
            + String(received_dtype)
        )

    var ndim = Int(dl_tensor.ndim)
    var shape = NDArrayShape(ndim=ndim, initialized=True)
    for i in range(ndim):
        shape[i] = Int(dl_tensor.shape[i])

    var strides: NDArrayStrides
    if dl_tensor.strides:
        strides = NDArrayStrides(ndim=ndim, initialized=True)
        for i in range(ndim):
            strides[i] = Int(dl_tensor.strides[i])
    else:
        strides = NDArrayStrides(shape, order="C")

    var data_ptr = dl_tensor.data.bitcast[Scalar[dtype]]()
    if dl_tensor.byte_offset > 0:
        data_ptr = data_ptr + Int(dl_tensor.byte_offset) // size_of[dtype]()

    var size = shape.size()
    var buf = DataContainer[dtype](
        ptr=data_ptr.unsafe_origin_cast[MutExternalOrigin](),
        size=size,
        copy=False,
    )

    var result = NDArray[dtype](shape, order="C")
    result._buf = buf^
    result.strides = strides

    return result^


def from_numpy[dtype: DType](array: PythonObject) raises -> NDArray[dtype]:
    """Imports a NumPy array into a NuMojo NDArray via zero-copy.

    This is a fast path specifically optimized for NumPy that uses the
    `__array_interface__` protocol to extract the data pointer directly,
    avoiding PyCapsule overhead entirely.
    This method is generally faster than using from_dlpack for NumPy arrays.

    Parameters:
        dtype: Expected data type of the array elements.

    Args:
        array: A NumPy ndarray object.

    Returns:
        A new NDArray that shares memory with the input array.

    Raises:
        Error: If the array is not on CPU (only CPU tensors are supported).

    Notes:
        - The imported array shares memory with the source. Modifications to
          either will be visible in both.
        - This uses NumPy's `__array_interface__` instead of the DLPack protocol.
        - Strides are converted from bytes to elements automatically.
    """
    var device_info = array.__dlpack_device__()
    var device_type = Int(py=device_info[0])
    if device_type != DLDevice.CPU:
        raise Error(
            "from_numpy: only CPU tensors are supported, got device type "
            + String(device_type)
        )

    var data_ptr = array.__array_interface__[PythonObject("data")][
        PythonObject(0)
    ].unsafe_get_as_pointer[dtype]()

    var ndim = Int(py=array.ndim)
    var shape = NDArrayShape(ndim=ndim, initialized=True)
    for i in range(ndim):
        shape[i] = Int(py=array.shape[i])

    var element_size = Int(py=array.itemsize)
    var strides = NDArrayStrides(ndim=ndim, initialized=True)
    for i in range(ndim):
        strides[i] = Int(py=array.strides[i]) // element_size

    var size = shape.size()
    var buf = DataContainer[dtype](
        ptr=data_ptr.unsafe_origin_cast[MutExternalOrigin](),
        size=size,
        copy=False,
    )

    var result = NDArray[dtype](shape, order="C")
    result._buf = buf^
    result.strides = strides

    return result^
