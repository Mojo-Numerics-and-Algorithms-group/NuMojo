from numojo.routines.creation import fromstring
from collections.optional import Optional
from python import Python, PythonObject
from memory import UnsafePointer, Span

# We call into the numpy backend for now, this at least let's people go back and forth smoothly.
# might consider implementing a funciton to write a .numojo file which can be read by both numpy and numojo.


fn load[
    dtype: DType = f64
](
    file: String,
    allow_pickle: Bool = False,
    fix_imports: Bool = True,
    encoding: String = "ASCII",
    *,
    max_header_size: Int = 10000,
) raises -> NDArray[dtype]:
    """
    Load arrays or pickled objects from .npy, .npz or pickled files.

    Args:
        file: The file to read. File-like objects must support the seek() and read() methods.
        allow_pickle: Allow loading pickled object arrays stored in npy files.
        fix_imports: Only useful when loading Python 2 generated pickled files on Python 3.
        encoding: What encoding to use when reading Python 2 strings.
        max_header_size: Maximum allowed size of the header.

    Returns:
        Data stored in the file.
    """
    var np = Python.import_module("numpy")
    var data = np.load(
        file=file,
        allow_pickle=allow_pickle,
        fix_imports=fix_imports,
        encoding=encoding,
        max_header_size=max_header_size,
    )
    var array = numojo.array[dtype](data=data)
    return array^


# @parameter
# fn _get_dtype_string[dtype: DType]() -> String:
#     """
#     Get the numpy-compatible dtype string for the given DType.

#     Parameters:
#         dtype: The DType to convert.

#     Returns:
#         A string representing the dtype in numpy format.
#     """

#     @parameter
#     if dtype == DType.bool:
#         return "'|b1'"
#     elif dtype == DType.int8:
#         return "'|i1'"
#     elif dtype == DType.int16:
#         return "'<i2'"
#     elif dtype == DType.int32:
#         return "'<i4'"
#     elif dtype == DType.int64:
#         return "'<i8'"
#     elif dtype == DType.uint8:
#         return "'|u1'"
#     elif dtype == DType.uint16:
#         return "'<u2'"
#     elif dtype == DType.uint32:
#         return "'<u4'"
#     elif dtype == DType.uint64:
#         return "'<u8'"
#     elif dtype == DType.float16:
#         return "'<f2'"
#     elif dtype == DType.float32:
#         return "'<f4'"
#     elif dtype == DType.float64:
#         return "'<f8'"
#     elif dtype == DType.int:
#         # Assuming index is 64-bit signed integer
#         return "'<i8'"
#     else:
#         return "'<f8'"


# fn _write_uint16_le(mut file: FileHandle, value: UInt16) raises:
#     """Write a 16-bit unsigned integer in little-endian format."""
#     var bytes_ptr = UnsafePointer[UInt8].alloc(2)
#     bytes_ptr[0] = UInt8(value & 0xFF)
#     bytes_ptr[1] = UInt8((value >> 8) & 0xFF)
#     var span = Span[UInt8](bytes_ptr, 2)
#     file.write_bytes(span)
#     bytes_ptr.free()


# fn savenpy[
#     dtype: DType = f64
# ](fname: String, array: NDArray[dtype], allow_pickle: Bool = True) raises:
#     """
#     Save an array to a binary file in NumPy .npy format.

#     This is a pure Mojo implementation that writes .npy files without using Python.
#     The file format follows the NumPy .npy specification v1.0.

#     Args:
#         fname: File or filename to which the data is saved. If fname is a string,
#                a .npy extension will be appended to the filename if it does not
#                already have one.
#         array: Array data to be saved.
#         allow_pickle: Allow saving object arrays using Python pickles.
#     """
#     # Add .npy extension if not present
#     var filename = fname
#     if not filename.endswith(".nmj"):
#         filename += ".nmj"

#     # Open file for binary writing
#     var file = open(filename, "wb")

#     try:
#         # Write magic string: \x93NUMPY (6 bytes)
#         var magic_ptr = UnsafePointer[UInt8].alloc(6)
#         magic_ptr[0] = 0x93  # \x93
#         magic_ptr[1] = ord("N")
#         magic_ptr[2] = ord("U")
#         magic_ptr[3] = ord("M")
#         magic_ptr[4] = ord("P")
#         magic_ptr[5] = ord("Y")
#         var magic_span = Span[UInt8](magic_ptr, 6)
#         file.write_bytes(magic_span)
#         magic_ptr.free()

#         # Write version: major=1, minor=0 (2 bytes)
#         var version_ptr = UnsafePointer[UInt8].alloc(2)
#         version_ptr[0] = 1  # major version
#         version_ptr[1] = 0  # minor version
#         var version_span = Span[UInt8](version_ptr, 2)
#         file.write_bytes(version_span)
#         version_ptr.free()

#         # Create header dictionary as string
#         var dtype_str = _get_dtype_string[dtype]()
#         var fortran_order = "True" if array.flags.F_CONTIGUOUS else "False"

#         # Build shape tuple string
#         var shape_str = String("(")
#         for i in range(array.ndim):
#             shape_str += String(array.shape[i])
#             if array.ndim == 1:
#                 shape_str += ","  # Single element tuple needs comma
#             elif i < array.ndim - 1:
#                 shape_str += ", "
#         shape_str += ")"

#         # Create header dictionary string
#         var header = "{'descr': " + dtype_str + ", 'fortran_order': " + fortran_order + ", 'shape': " + shape_str + ", }"

#         # Pad header to be divisible by 64 for alignment
#         var base_size = 6 + 2 + 2  # magic + version + header_len
#         var header_with_newline = header + "\n"
#         var total_size = base_size + len(header_with_newline)
#         var padding_needed = (64 - (total_size % 64)) % 64

#         # Add padding spaces
#         for _ in range(padding_needed):
#             header_with_newline = (
#                 header_with_newline[:-1] + " \n"
#             )  # Insert space before newline

#         # Write header length (2 bytes, little-endian)
#         var final_header_len = UInt16(len(header_with_newline))
#         _write_uint16_le(file, final_header_len)

#         # Write header as bytes
#         var header_bytes = header_with_newline.as_bytes()
#         var header_ptr = UnsafePointer[UInt8].alloc(len(header_bytes))
#         for i in range(len(header_bytes)):
#             header_ptr[i] = header_bytes[i]
#         var header_span = Span[UInt8](header_ptr, len(header_bytes))
#         file.write_bytes(header_span)
#         header_ptr.free()

#         # Write array data
#         var data_size = array.size * dtype.sizeof()
#         var data_ptr = array._buf.ptr.bitcast[UInt8]()
#         var data_span = Span[UInt8](data_ptr, data_size)
#         file.write_bytes(data_span)

#     finally:
#         file.close()


fn save[
    dtype: DType = f64
](fname: String, array: NDArray[dtype], allow_pickle: Bool = True,) raises:
    """
    Save an array to a binary file in NumPy .npy format.

    Args:
        fname: File or filename to which the data is saved.
        array: Array data to be saved.
        allow_pickle: Allow saving object arrays using Python pickles.
    """
    var np = Python.import_module("numpy")
    var np_arr = array.to_numpy()
    np.save(
        file=fname,
        arr=np_arr,
        allow_pickle=allow_pickle,
    )


fn loadtxt[
    dtype: DType = f64
](
    fname: String,
    comments: String = "#",
    delimiter: String = " ",
    skiprows: Int = 0,
    ndmin: Int = 0,
) raises -> NDArray[dtype]:
    """
    Load data from a text file.

    Args:
        fname: File, filename, list, or generator to read.
        comments: The characters or list of characters used to indicate the start of a comment.
        delimiter: The string used to separate values.
        skiprows: Skip the first skiprows lines.
        ndmin: The returned array will have at least ndmin dimensions.

    Returns:
        Data read from the text file.
    """
    var np = Python.import_module("numpy")
    var data = np.loadtxt(
        fname=fname,
        comments=comments,
        delimiter=delimiter,
        skiprows=skiprows,
        ndmin=ndmin,
    )
    var array = numojo.array[dtype](data=data)
    return array^


fn savetxt[
    dtype: DType = f64
](
    fname: String,
    array: NDArray[dtype],
    fmt: String = "%.18e",
    delimiter: String = " ",
    newline: String = "\n",
    header: String = "",
    footer: String = "",
    comments: String = "#",
) raises:
    """
    Save an array to a text file.

    Args:
        fname: If the filename ends in .gz, the file is automatically saved in compressed gzip format.
        array: 1D or 2D array_like data to be saved to a text file.
        fmt: A single format (%10.5f), a sequence of formats, or a multi-format string.
        delimiter: String or character separating columns.
        newline: String or character separating lines.
        header: String that will be written at the beginning of the file.
        footer: String that will be written at the end of the file.
        comments: String that will be prepended to the header and footer strings.
    """
    var np = Python.import_module("numpy")
    var np_arr = array.to_numpy()
    np.savetxt(
        fname=fname,
        X=np_arr,
        fmt=fmt,
        delimiter=delimiter,
        newline=newline,
        header=header,
        footer=footer,
        comments=comments,
    )
