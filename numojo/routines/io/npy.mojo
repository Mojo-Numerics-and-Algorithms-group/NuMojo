# ===----------------------------------------------------------------------=== #
# NuMojo: NumPy .npy format
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
NumPy .npy format (numojo.routines.io.npy).
===========================================
Pure-Mojo reader and writer for NumPy's `.npy` array format.

`.npy` is a documented, self-contained binary format: a short magic prelude
followed by an ASCII header describing dtype/shape/order, followed by the raw
payload. This is distinct from `numojo.routines.io.files.save`/`load`, which route
through Python's `numpy.save`/`numpy.load` and therefore require Python.

Exports
-------
- `save_npy`: Write an NDArray to a `.npy` file.
- `load_npy`: Read an NDArray from a `.npy` file.

Notes
-----
- Only dtypes with a direct NumPy equivalent can round-trip: `bool`, the
  signed/unsigned integers up to 64 bits, and `float16`/`float32`/`float64`.
  NuMojo-only widths (`int128`, `int256`, `uint128`, `uint256`) and
  `bfloat16` have no NumPy dtype, so both directions raise for them.
- `save_npy` always writes format version 1.0 in C order (row-major); the
  source array is made contiguous first if it is not already.
- `load_npy` accepts version 1.0 and 2.0 headers, and both C- and
  Fortran-ordered payloads (the returned array matches the file's order).
  It rejects big-endian dtypes (`.npy` written on a big-endian machine) since
  nothing here byte-swaps.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.collections.span import Span
from std.memory import bitcast
from std.sys.info import size_of

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray

# The `.npy` prelude is aligned to a multiple of this many bytes.
comptime _NPY_ALIGN = 64

comptime _NPY_MAGIC_BYTE = UInt8(0x93)
"""First byte of the `.npy` magic. Not valid standalone UTF-8, so it is kept
apart from the ASCII tail `NUMPY` and assembled by `_npy_magic`."""


def _npy_magic() -> List[UInt8]:
    """The 6-byte `.npy` magic: `0x93` followed by `NUMPY`."""
    var out = List[UInt8](capacity=6)
    out.append(_NPY_MAGIC_BYTE)
    for c in String("NUMPY").as_bytes():
        out.append(c)
    return out^


def _npy_convert_dtype[dtype: DType]() -> String:
    """NumPy's `descr` string for `dtype`, e.g. `'<f4'` for `float32`.

    Single-byte dtypes carry `'|'` (byte order does not apply to them),
    matching what `numpy.save` writes. Returns `""` when `dtype` has no
    NumPy equivalent.
    """

    comptime if dtype == DType.bool:
        return "|b1"
    elif dtype == DType.int8:
        return "|i1"
    elif dtype == DType.int16:
        return "<i2"
    elif dtype == DType.int32:
        return "<i4"
    elif dtype == DType.int64 or dtype == DType.int:
        return "<i8"
    elif dtype == DType.uint8:
        return "|u1"
    elif dtype == DType.uint16:
        return "<u2"
    elif dtype == DType.uint32:
        return "<u4"
    elif dtype == DType.uint64:
        return "<u8"
    elif dtype == DType.float16:
        return "<f2"
    elif dtype == DType.float32:
        return "<f4"
    elif dtype == DType.float64:
        return "<f8"
    else:
        return ""


def _same_npy_dtype_layout(file_descr: String, expected: String) -> Bool:
    """Whether `file_descr` (from a file's header) names the same on-disk
    layout as `expected` (computed by `_npy_convert_dtype`, always in `'<'`
    little-endian form).

    The kind-and-itemsize suffix (e.g. `f8`) must match exactly; only the
    leading byte-order character is allowed to differ, and only to another
    spelling of "little-endian": `'='` ("native", which is little-endian on
    every platform NuMojo targets), or, when the itemsize is one byte,
    `'|'` (order does not apply, since there is nothing to order).
    """
    if file_descr == expected:
        return True
    if file_descr.byte_length() != 3 or expected.byte_length() != 3:
        return False

    var fbytes = file_descr.as_bytes()
    var ebytes = expected.as_bytes()

    var suffix_matches = (fbytes[1] == ebytes[1]) and (fbytes[2] == ebytes[2])
    if not suffix_matches:
        return False

    var acceptable_orders: List[UInt8] = [UInt8(ord("="))]
    if fbytes[2] == UInt8(ord("1")):
        acceptable_orders.append(UInt8(ord("|")))

    for candidate in acceptable_orders:
        if fbytes[0] == candidate:
            return True
    return False


def _shape_tuple_text(A: NDArray, ndim: Int) raises -> String:
    """`A.shape` rendered as a Python tuple literal, e.g. `(2, 3)`.

    Rank 1 carries a trailing comma (`(3,)`) because `(3)` is a plain `int`
    literal in Python, not a 1-tuple, and NumPy's header is Python source
    text. Rank 0 renders as `()`, NumPy's spelling for a 0-d array.
    """
    var out = String("(")
    for i in range(ndim):
        out += String(Int(A.shape[i]))
        if i != ndim - 1:
            out += ", "
    if ndim == 1:
        out += ","
    out += ")"
    return out^


def _npy_header_dict_value(header: String, key: String) raises -> String:
    """The value text for `key` in a `.npy` header dict.

    The header's grammar only ever pairs three fixed keys with three fixed
    value shapes, so each is read by its own shape rather than by a generic
    bracket-depth scan: `descr`'s value is a quoted string, `shape`'s is a
    parenthesized tuple (whose inner commas must not be mistaken for the
    entry separator), and `fortran_order`'s is the bare word `True`/`False`.

    Returns:
        `descr`: the string with its quotes stripped, e.g. `<f8`.
        `shape`: the tuple text including its parentheses, e.g. `(2, 3)`.
        `fortran_order`: the bare token, e.g. `False`.
    """
    var key_pos = header.find(String("'", key, "'"))
    if key_pos < 0:
        raise Error(
            NumojoError(
                category="value",
                message=String("`.npy` header has no '{}' key.").format(key),
                location="npy._npy_header_dict_value()",
            )
        )
    var colon_pos = header.find(":", key_pos)
    if colon_pos < 0:
        raise Error(
            NumojoError(
                category="value",
                message=String("Malformed '{}' entry in .npy header.").format(
                    key
                ),
                location="npy._npy_header_dict_value()",
            )
        )

    var bytes = header.as_bytes()
    var value_start = colon_pos + 1
    while value_start < len(bytes) and bytes[value_start] == UInt8(ord(" ")):
        value_start += 1
    if value_start >= len(bytes):
        raise Error(
            NumojoError(
                category="value",
                message=String("Malformed '{}' entry in .npy header.").format(
                    key
                ),
                location="npy._npy_header_dict_value()",
            )
        )

    var opener = bytes[value_start]
    var closer: UInt8
    var strip_opener = False
    if opener == UInt8(ord("'")):
        closer = UInt8(ord("'"))
        strip_opener = True
    elif opener == UInt8(ord("(")):
        closer = UInt8(ord(")"))
    else:
        # Bare token (`fortran_order`'s `True`/`False`): runs to the next
        # top-level comma instead of a matching closer.
        closer = UInt8(ord(","))

    var value_end = value_start + 1
    while value_end < len(bytes) and bytes[value_end] != closer:
        value_end += 1
    if value_end >= len(bytes) and closer != UInt8(ord(",")):
        raise Error(
            NumojoError(
                category="value",
                message=String("Malformed '{}' entry in .npy header.").format(
                    key
                ),
                location="npy._npy_header_dict_value()",
            )
        )

    if strip_opener:
        return String(
            StringSlice(
                unsafe_from_utf8=Span(bytes)[value_start + 1 : value_end]
            )
        )
    elif closer == UInt8(ord(",")):
        return String(
            String(
                StringSlice(unsafe_from_utf8=Span(bytes)[value_start:value_end])
            ).strip()
        )
    else:
        # Include the parentheses -- `_parse_shape_tuple` wants the full
        # tuple text.
        return String(
            StringSlice(
                unsafe_from_utf8=Span(bytes)[value_start : value_end + 1]
            )
        )


def _parse_shape_tuple(text: String) raises -> List[Int]:
    """The dimensions in a `.npy` header's `shape` tuple text, e.g. `(2, 3)`.

    Strips the surrounding parentheses and splits on commas -- the same
    grammar a rank-1 tuple's trailing comma (`(3,)`) produces an empty
    trailing segment for, which is dropped along with any other blank
    segment. `()` (NumPy's rank-0 scalar) parses to an empty list.
    """
    var inner: String = String(text.strip())
    if inner.startswith("("):
        var without_open = String(inner[byte = 1 : inner.byte_length()])
        inner = without_open
    if inner.endswith(")"):
        var without_close = String(inner[byte = 0 : inner.byte_length() - 1])
        inner = without_close

    var out = List[Int]()
    for segment in inner.split(","):
        var trimmed = String(segment).strip()
        if trimmed.byte_length() > 0:
            out.append(Int(trimmed))
    return out^


def _append_scalar_bytes[
    dtype: DType
](val: Scalar[dtype], mut out: List[UInt8]):
    """Appends `val`'s little-endian byte representation to `out`.

    Goes through a value-level `bitcast` (SIMD lane reinterpretation) rather
    than a raw pointer cast, so it is correct regardless of a scalar's
    underlying alignment.
    """
    comptime n = size_of[Scalar[dtype]]()
    var b = bitcast[DType.uint8, n](val)

    comptime for j in range(n):
        out.append(b[j])


def _read_scalar_bytes[
    dtype: DType
](data: List[UInt8], offset: Int) -> Scalar[dtype]:
    """The scalar whose little-endian bytes are `data[offset:offset+n]`."""
    comptime n = size_of[Scalar[dtype]]()
    var b = SIMD[DType.uint8, n]()

    comptime for j in range(n):
        b[j] = data[offset + j]
    return bitcast[dtype, 1](b)


def save_npy[dtype: DType](path: String, A: NDArray[dtype]) raises:
    """
    Writes an array to a `.npy` file (NumPy format version 1.0), with no
    Python involved.

    The bytes are what `numpy.save` would write for the same array (C
    order), so `numpy.load(path)` reads it back with the right dtype and
    shape.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        path: File path to write to.
        A: The array to save.

    Raises:
        NumojoError: If `dtype` has no NumPy equivalent (`int128`, `int256`,
            `uint128`, `uint256`, or `bfloat16`).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.io.npy import save_npy, load_npy

        var a = nm.arange[nm.f32](6).reshape(nm.Shape(2, 3))
        save_npy("a.npy", a)
        var b = load_npy[nm.f32]("a.npy")
        ```
    """
    comptime descr = _npy_convert_dtype[dtype]()
    comptime if descr == "":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "{} has no NumPy dtype, so it has no .npy representation."
                ).format(String(dtype)),
                location="npy.save_npy()",
            )
        )

    var Ac = A.contiguous()

    var dict_text = String(
        "{'descr': '",
        descr,
        "', 'fortran_order': False, 'shape': ",
        _shape_tuple_text(Ac, Ac.ndim),
        ", }",
    )

    # 6 magic + 2 version + 2 header-length bytes = 10-byte prelude. Pad the
    # dict (plus trailing newline) so the whole prelude is a multiple of
    # `_NPY_ALIGN`.
    var pad = _NPY_ALIGN - ((10 + dict_text.byte_length() + 1) % _NPY_ALIGN)
    var header = dict_text
    for _ in range(pad):
        header += " "
    header += "\n"

    if header.byte_length() > 65535:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "`.npy` header is too large for format version 1.0"
                    " (shape has too many dimensions)."
                ),
                location="npy.save_npy()",
            )
        )

    var out = _npy_magic()
    out.append(UInt8(1))  # major version
    out.append(UInt8(0))  # minor version
    var header_len = header.byte_length()
    out.append(UInt8(header_len & 0xFF))
    out.append(UInt8((header_len >> 8) & 0xFF))
    for c in header.as_bytes():
        out.append(c)

    var f = open(path, "w")
    f.write_bytes(Span(out))

    var payload = List[UInt8](capacity=Ac.size * size_of[Scalar[dtype]]())
    for i in range(Ac.size):
        _append_scalar_bytes[dtype](Ac.unsafe_get(i), payload)
    if len(payload) > 0:
        f.write_bytes(Span(payload))
    f.close()


def load_npy[dtype: DType](path: String) raises -> NDArray[dtype]:
    """
    Reads an array from a `.npy` file written by `numpy.save` (or by
    `save_npy`), with no Python involved.

    Parameters:
        dtype: Expected datatype of the NDArray elements.

    Args:
        path: File path to read from.

    Returns:
        The array stored in the file, in whichever order (`C` or `F`) it
        was written in.

    Raises:
        NumojoError: If `dtype` has no NumPy equivalent; if the file is not
            a `.npy` file, is truncated, or uses an unsupported format
            version; if the file's dtype does not match `dtype` (including
            big-endian files, which nothing here byte-swaps); or if the
            payload size disagrees with the header.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.io.npy import load_npy

        var a = load_npy[nm.f32]("a.npy")
        ```
    """
    comptime expected_descr = _npy_convert_dtype[dtype]()
    comptime if expected_descr == "":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "{} has no NumPy dtype, so no .npy file can hold it."
                ).format(String(dtype)),
                location="npy.load_npy()",
            )
        )

    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()

    var magic = _npy_magic()
    if len(data) < 10:
        raise Error(
            NumojoError(
                category="value",
                message="File is too short to be a .npy file.",
                location="npy.load_npy()",
            )
        )
    for i in range(len(magic)):
        if data[i] != magic[i]:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Bad magic bytes -- not a .npy file. A .npz archive"
                        " is a zip container, not a .npy file: unzip it, or"
                        " re-save each array with numpy.save."
                    ),
                    location="npy.load_npy()",
                )
            )

    var major = Int(data[6])
    if major != 1 and major != 2:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "Unsupported .npy format version {} -- only versions 1.x"
                    " and 2.x are understood."
                ).format(major),
                location="npy.load_npy()",
            )
        )

    var header_len: Int
    var header_start: Int
    if major == 1:
        header_len = Int(data[8]) | (Int(data[9]) << 8)
        header_start = 10
    else:
        if len(data) < 12:
            raise Error(
                NumojoError(
                    category="value",
                    message="Truncated .npy version-2 header.",
                    location="npy.load_npy()",
                )
            )
        header_len = (
            Int(data[8])
            | (Int(data[9]) << 8)
            | (Int(data[10]) << 16)
            | (Int(data[11]) << 24)
        )
        header_start = 12
    if len(data) < header_start + header_len:
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .npy header.",
                location="npy.load_npy()",
            )
        )

    var header = String(
        StringSlice(
            unsafe_from_utf8=Span(data)[
                header_start : header_start + header_len
            ]
        )
    )

    var file_descr = _npy_header_dict_value(header, "descr")
    if not _same_npy_dtype_layout(file_descr, expected_descr):
        if file_descr.startswith(">"):
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Big-endian .npy (descr '{}') -- nothing here"
                        " byte-swaps; re-save it native, e.g. with"
                        " arr.astype(arr.dtype.newbyteorder('<'))."
                    ).format(file_descr),
                    location="npy.load_npy()",
                )
            )
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "dtype mismatch -- the file holds '{}' but '{}' was"
                    " requested."
                ).format(file_descr, expected_descr),
                location="npy.load_npy()",
            )
        )

    var fortran_order = _npy_header_dict_value(header, "fortran_order")
    var order = "F" if fortran_order == "True" else "C"

    var dims = _parse_shape_tuple(_npy_header_dict_value(header, "shape"))
    var result = NDArray[dtype](NDArrayShape(dims), order=order)

    var offset = header_start + header_len
    var nbytes = result.size * size_of[Scalar[dtype]]()
    if len(data) - offset != nbytes:
        raise Error(
            NumojoError(
                category="value",
                message="Payload size doesn't match the .npy header.",
                location="npy.load_npy()",
            )
        )

    var item_size = size_of[Scalar[dtype]]()
    for i in range(result.size):
        result.unsafe_set(
            i, _read_scalar_bytes[dtype](data, offset + i * item_size)
        )

    return result^
