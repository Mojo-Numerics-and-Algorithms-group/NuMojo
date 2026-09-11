# ===----------------------------------------------------------------------=== #
# NuMojo: PyTorch .pt format
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
PyTorch .pt format (numojo.routines.io.torch).
==============================================
Pure-Mojo reader and writer for a single plain tensor saved with `torch.save`.

A `.pt` file is a ZIP archive holding a pickled object graph plus the
tensor's raw storage. This module recognizes only the fixed pickle shape a
bare `torch.save(tensor, path)` produces, not general pickle.

Exports
-------
- `save_torch`: Write an NDArray to a `.pt` file as a single tensor.
- `load_torch`: Read an NDArray from a `.pt` file holding a single tensor.

Notes
-----
- Only a file holding exactly one plain `torch.Tensor` is understood; a
  state dict, a full model, or any other pickled object raises.
- Only dtypes with a direct PyTorch storage equivalent can round-trip:
  `bool`, the signed integers up to 64 bits, `uint8`, and
  `float16`/`bfloat16`/`float32`/`float64`.
- `load_torch` only reads `'cpu'` tensors; it raises for CUDA/MPS.
- `requires_grad` and the autograd graph are not represented by `NDArray`;
  `save_torch` always writes `requires_grad=False`.
- Only the ZIP-archive `.pt` format (PyTorch's default since 1.6) is
  understood; ZIP64 and the legacy bare-pickle format are not.
- 0-d (scalar) tensors raise: NuMojo's `NDArray` has no rank-0 support yet.
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

# ===----------------------------------------------------------------------=== #
# Scalar <-> raw little-endian bytes
# ===----------------------------------------------------------------------=== #


def _same_width_uint_dtype[n: Int]() -> DType:
    """The unsigned integer `DType` occupying exactly `n` bytes.

    `bitcast` refuses a couple of direct float16 <-> 2x-uint8 shapes on
    non-GPU targets, so a byte buffer is bounced through this same-width
    scalar unsigned type on the way to/from a narrow float instead.
    """
    comptime if n == 1:
        return DType.uint8
    elif n == 2:
        return DType.uint16
    elif n == 4:
        return DType.uint32
    elif n == 8:
        return DType.uint64
    elif n == 16:
        return DType.uint128
    else:
        return DType.uint256


def _append_scalar_bytes[
    dtype: DType
](val: Scalar[dtype], mut out: List[UInt8]):
    """Appends `val`'s little-endian byte representation to `out`, via a
    value-level `bitcast` (SIMD lane reinterpretation, not a pointer cast).

    `bool` is handled separately: its packed `i1` backing crashes `bitcast`
    outright (not merely rejected -- a compiler pass failure), so its one
    storage byte is written directly from the boolean value instead.
    """
    comptime if dtype == DType.bool:
        out.append(UInt8(1) if Bool(val) else UInt8(0))
        return

    comptime n = size_of[Scalar[dtype]]()
    comptime u = _same_width_uint_dtype[n]()
    var as_uint = bitcast[u, 1](val)
    var b = bitcast[DType.uint8, n](as_uint)

    comptime for j in range(n):
        out.append(b[j])


def _read_scalar_bytes[
    dtype: DType
](data: List[UInt8], offset: Int) -> Scalar[dtype]:
    """The scalar whose little-endian bytes are `data[offset:offset+n]`.

    See `_append_scalar_bytes` for why `bool` skips `bitcast` entirely.
    """
    comptime if dtype == DType.bool:
        return rebind[Scalar[dtype]](
            SIMD[DType.bool, 1](data[offset] != UInt8(0))
        )

    comptime n = size_of[Scalar[dtype]]()
    comptime u = _same_width_uint_dtype[n]()
    var b = SIMD[DType.uint8, n]()

    comptime for j in range(n):
        b[j] = data[offset + j]
    var as_uint = bitcast[u, 1](b)
    return bitcast[dtype, 1](as_uint)


# ===----------------------------------------------------------------------=== #
# dtype <-> PyTorch storage class name
# ===----------------------------------------------------------------------=== #


def _torch_storage_kind[dtype: DType]() -> String:
    """The `<Kind>` in PyTorch's `torch.<Kind>Storage` for `dtype`, e.g.
    `"Float"` for `float32`. Returns `""` when `dtype` has no PyTorch
    storage type.
    """
    comptime if dtype == DType.float64:
        return "Double"
    elif dtype == DType.float32:
        return "Float"
    elif dtype == DType.float16:
        return "Half"
    elif dtype == DType.bfloat16:
        return "BFloat16"
    elif dtype == DType.int64 or dtype == DType.int:
        return "Long"
    elif dtype == DType.int32:
        return "Int"
    elif dtype == DType.int16:
        return "Short"
    elif dtype == DType.int8:
        return "Char"
    elif dtype == DType.uint8:
        return "Byte"
    elif dtype == DType.bool:
        return "Bool"
    else:
        return ""


# ===----------------------------------------------------------------------=== #
# Little-endian integer <-> bytes
# ===----------------------------------------------------------------------=== #


def _put_u16(mut out: List[UInt8], val: Int):
    out.append(UInt8(val & 0xFF))
    out.append(UInt8((val >> 8) & 0xFF))


def _put_u32(mut out: List[UInt8], val: Int):
    out.append(UInt8(val & 0xFF))
    out.append(UInt8((val >> 8) & 0xFF))
    out.append(UInt8((val >> 16) & 0xFF))
    out.append(UInt8((val >> 24) & 0xFF))


def _get_u16(data: List[UInt8], pos: Int) -> Int:
    return Int(data[pos]) | (Int(data[pos + 1]) << 8)


def _get_u32(data: List[UInt8], pos: Int) -> Int:
    return (
        Int(data[pos])
        | (Int(data[pos + 1]) << 8)
        | (Int(data[pos + 2]) << 16)
        | (Int(data[pos + 3]) << 24)
    )


# ===----------------------------------------------------------------------=== #
# CRC-32 (ISO 3309 / ZIP's checksum)
# ===----------------------------------------------------------------------=== #


def _crc32(data: List[UInt8]) -> UInt32:
    """The CRC-32 of `data`, computed bit by bit against the standard
    0xEDB88320 reflected polynomial (the one ZIP and gzip use)."""
    var crc: UInt32 = 0xFFFFFFFF
    for byte in data:
        crc = crc ^ UInt32(byte)
        for _ in range(8):
            var lsb_set = (crc & UInt32(1)) != UInt32(0)
            crc = crc >> 1
            if lsb_set:
                crc = crc ^ UInt32(0xEDB88320)
    return crc ^ 0xFFFFFFFF


# ===----------------------------------------------------------------------=== #
# ZIP container (STORED entries only -- what `torch.save` writes)
# ===----------------------------------------------------------------------=== #

comptime _ZIP_LOCAL_SIG = 0x04034B50
comptime _ZIP_CENTRAL_SIG = 0x02014B50
comptime _ZIP_EOCD_SIG = 0x06054B50


def _zip_write_stored(
    names: List[String], payloads: List[List[UInt8]]
) raises -> List[UInt8]:
    """A minimal ZIP archive holding `names[i]: payloads[i]` as uncompressed
    (STORED) entries -- the smallest ZIP `torch.load`'s own reader (and any
    standard unzip tool) accepts.
    """
    if len(names) != len(payloads):
        raise Error(
            NumojoError(
                category="value",
                message="`names` and `payloads` must have the same length.",
                location="torch._zip_write_stored()",
            )
        )

    var out = List[UInt8]()
    var local_offsets = List[Int]()

    for i in range(len(names)):
        local_offsets.append(len(out))
        var name_bytes = names[i].as_bytes()
        var payload = payloads[i].copy()
        var crc = _crc32(payload)

        _put_u32(out, _ZIP_LOCAL_SIG)
        _put_u16(out, 20)  # version needed to extract
        _put_u16(out, 0)  # general purpose bit flag
        _put_u16(out, 0)  # compression method: stored
        _put_u16(out, 0)  # last mod file time
        _put_u16(out, 0x21)  # last mod file date (1980-01-01)
        _put_u32(out, Int(crc))
        _put_u32(out, len(payload))  # compressed size
        _put_u32(out, len(payload))  # uncompressed size
        _put_u16(out, len(name_bytes))
        _put_u16(out, 0)  # extra field length
        for c in name_bytes:
            out.append(c)
        for b in payload:
            out.append(b)

    var central_start = len(out)
    for i in range(len(names)):
        var name_bytes = names[i].as_bytes()
        var payload = payloads[i].copy()
        var crc = _crc32(payload)

        _put_u32(out, _ZIP_CENTRAL_SIG)
        _put_u16(out, 20)  # version made by
        _put_u16(out, 20)  # version needed to extract
        _put_u16(out, 0)  # general purpose bit flag
        _put_u16(out, 0)  # compression method: stored
        _put_u16(out, 0)  # last mod file time
        _put_u16(out, 0x21)  # last mod file date
        _put_u32(out, Int(crc))
        _put_u32(out, len(payload))  # compressed size
        _put_u32(out, len(payload))  # uncompressed size
        _put_u16(out, len(name_bytes))
        _put_u16(out, 0)  # extra field length
        _put_u16(out, 0)  # file comment length
        _put_u16(out, 0)  # disk number start
        _put_u16(out, 0)  # internal file attributes
        _put_u32(out, 0)  # external file attributes
        _put_u32(out, local_offsets[i])  # relative offset of local header
        for c in name_bytes:
            out.append(c)

    var central_size = len(out) - central_start

    _put_u32(out, _ZIP_EOCD_SIG)
    _put_u16(out, 0)  # number of this disk
    _put_u16(out, 0)  # disk where central directory starts
    _put_u16(out, len(names))  # central directory records on this disk
    _put_u16(out, len(names))  # total central directory records
    _put_u32(out, central_size)
    _put_u32(out, central_start)
    _put_u16(out, 0)  # comment length

    return out^


def _zip_find_eocd(data: List[UInt8]) raises -> Int:
    """The byte offset of the end-of-central-directory record's signature.

    The record sits at the very end of the file unless a trailing comment
    follows it (max 65535 bytes), so the search only has to cover that
    trailing window, scanning backward so a signature-shaped byte sequence
    inside the comment can't be mistaken for the real one.
    """
    var window_start = 0
    if len(data) > 65557:
        window_start = len(data) - 65557

    var pos = len(data) - 22
    while pos >= window_start:
        if (
            data[pos] == UInt8(0x50)
            and data[pos + 1] == UInt8(0x4B)
            and data[pos + 2] == UInt8(0x05)
            and data[pos + 3] == UInt8(0x06)
        ):
            return pos
        pos -= 1

    raise Error(
        NumojoError(
            category="value",
            message=(
                "No end-of-central-directory record found -- not a .pt"
                " (ZIP) file, or the file is truncated."
            ),
            location="torch._zip_find_eocd()",
        )
    )


def _zip_extract(data: List[UInt8], name_suffix: String) raises -> List[UInt8]:
    """The uncompressed bytes of the one ZIP entry whose name ends with
    `name_suffix` (entries live under an archive-name folder this doesn't
    otherwise care about, e.g. `archive/data.pkl`).
    """
    var eocd = _zip_find_eocd(data)
    var record_count = _get_u16(data, eocd + 10)
    var central_offset = _get_u32(data, eocd + 16)

    if record_count == 0xFFFF or central_offset == 0xFFFFFFFF:
        raise Error(
            NumojoError(
                category="value",
                message="ZIP64 archives are not supported.",
                location="torch._zip_extract()",
            )
        )

    var pos = central_offset
    for _ in range(record_count):
        if (
            data[pos] != UInt8(0x50)
            or data[pos + 1] != UInt8(0x4B)
            or data[pos + 2] != UInt8(0x01)
            or data[pos + 3] != UInt8(0x02)
        ):
            raise Error(
                NumojoError(
                    category="value",
                    message="Malformed ZIP central directory entry.",
                    location="torch._zip_extract()",
                )
            )
        var method = _get_u16(data, pos + 10)
        var comp_size = _get_u32(data, pos + 20)
        var name_len = _get_u16(data, pos + 28)
        var extra_len = _get_u16(data, pos + 30)
        var comment_len = _get_u16(data, pos + 32)
        var local_offset = _get_u32(data, pos + 42)
        var name = String(
            StringSlice(
                unsafe_from_utf8=Span(data)[pos + 46 : pos + 46 + name_len]
            )
        )

        if name.endswith(name_suffix):
            if method != 0:
                raise Error(
                    NumojoError(
                        category="value",
                        message=String(
                            "Entry '{}' is compressed -- only uncompressed"
                            " (STORED) .pt archives are supported."
                        ).format(name),
                        location="torch._zip_extract()",
                    )
                )
            var local_name_len = _get_u16(data, local_offset + 26)
            var local_extra_len = _get_u16(data, local_offset + 28)
            var data_start = (
                local_offset + 30 + local_name_len + local_extra_len
            )
            var out = List[UInt8](capacity=comp_size)
            for i in range(comp_size):
                out.append(data[data_start + i])
            return out^

        pos += 46 + name_len + extra_len + comment_len

    raise Error(
        NumojoError(
            category="value",
            message=String(
                "No entry ending in '{}' found in the .pt archive."
            ).format(name_suffix),
            location="torch._zip_extract()",
        )
    )


# ===----------------------------------------------------------------------=== #
# Pickle stream: read/write exactly one `_rebuild_tensor_v2(...)` call
# ===----------------------------------------------------------------------=== #

comptime _PICKLE_PROTO = UInt8(0x80)
comptime _PICKLE_GLOBAL = UInt8(0x63)  # 'c'
comptime _PICKLE_BINPUT = UInt8(0x71)  # 'q'
comptime _PICKLE_LONG_BINPUT = UInt8(0x72)  # 'r'
comptime _PICKLE_MARK = UInt8(0x28)  # '('
comptime _PICKLE_BINUNICODE = UInt8(0x58)  # 'X'
comptime _PICKLE_TUPLE = UInt8(0x74)  # 't'
comptime _PICKLE_BINPERSID = UInt8(0x51)  # 'Q'
comptime _PICKLE_BININT1 = UInt8(0x4B)  # 'K'
comptime _PICKLE_BININT2 = UInt8(0x4D)  # 'M'
comptime _PICKLE_BININT = UInt8(0x4A)  # 'J'
comptime _PICKLE_LONG1 = UInt8(0x8A)
comptime _PICKLE_TUPLE1 = UInt8(0x85)
comptime _PICKLE_TUPLE2 = UInt8(0x86)
comptime _PICKLE_TUPLE3 = UInt8(0x87)
comptime _PICKLE_EMPTY_TUPLE = UInt8(0x29)  # ')'
comptime _PICKLE_NEWTRUE = UInt8(0x88)
comptime _PICKLE_NEWFALSE = UInt8(0x89)
comptime _PICKLE_REDUCE = UInt8(0x52)  # 'R'
comptime _PICKLE_STOP = UInt8(0x2E)  # '.'


def _pickle_expect(
    data: List[UInt8], mut pos: Int, op: UInt8, what: String
) raises:
    if pos >= len(data) or data[pos] != op:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Unexpected .pt pickle opcode while reading {} at byte"
                    " {}. load_torch only reads the fixed opcode shape a"
                    " plain torch.save(tensor, path) produces."
                ).format(what, pos),
                location="torch._pickle_expect()",
            )
        )
    pos += 1


def _pickle_skip_binput(data: List[UInt8], mut pos: Int):
    """Skips a memo-writing opcode (`BINPUT`/`LONG_BINPUT`) if present --
    real `torch.save` output has one after nearly every pushed value, this
    module's own writer emits none, and neither carries information this
    reader needs.
    """
    if pos < len(data) and data[pos] == _PICKLE_BINPUT:
        pos += 2
    elif pos < len(data) and data[pos] == _PICKLE_LONG_BINPUT:
        pos += 5


def _pickle_read_binint(data: List[UInt8], mut pos: Int) raises -> Int:
    if pos >= len(data):
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .pt pickle stream while reading an int.",
                location="torch._pickle_read_binint()",
            )
        )
    var op = data[pos]
    pos += 1
    if op == _PICKLE_BININT1:
        var v = Int(data[pos])
        pos += 1
        return v
    elif op == _PICKLE_BININT2:
        var v = _get_u16(data, pos)
        pos += 2
        return v
    elif op == _PICKLE_BININT:
        var v = _get_u32(data, pos)
        pos += 4
        if v >= 0x80000000:
            v -= 0x100000000
        return v
    elif op == _PICKLE_LONG1:
        var n = Int(data[pos])
        pos += 1
        var v = 0
        for i in range(n):
            v |= Int(data[pos + i]) << (8 * i)
        pos += n
        return v
    else:
        raise Error(
            NumojoError(
                category="value",
                message="Expected an integer opcode in the .pt pickle stream.",
                location="torch._pickle_read_binint()",
            )
        )


def _pickle_read_line(data: List[UInt8], mut pos: Int) raises -> String:
    """The text from `pos` up to (and past) the next `\\n` -- `GLOBAL`'s
    module and qualified-name fields are newline-terminated even in a
    binary-protocol pickle.
    """
    var start = pos
    while pos < len(data) and data[pos] != UInt8(ord("\n")):
        pos += 1
    if pos >= len(data):
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .pt pickle stream while reading a GLOBAL.",
                location="torch._pickle_read_line()",
            )
        )
    var s = String(StringSlice(unsafe_from_utf8=Span(data)[start:pos]))
    pos += 1
    return s^


def _pickle_read_global(
    data: List[UInt8], mut pos: Int
) raises -> Tuple[String, String]:
    _pickle_expect(data, pos, _PICKLE_GLOBAL, "a class/function reference")
    var module = _pickle_read_line(data, pos)
    var qualname = _pickle_read_line(data, pos)
    return (module^, qualname^)


def _pickle_read_binunicode(data: List[UInt8], mut pos: Int) raises -> String:
    _pickle_expect(data, pos, _PICKLE_BINUNICODE, "a string")
    if pos + 4 > len(data):
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .pt pickle stream while reading a string.",
                location="torch._pickle_read_binunicode()",
            )
        )
    var n = _get_u32(data, pos)
    pos += 4
    var s = String(StringSlice(unsafe_from_utf8=Span(data)[pos : pos + n]))
    pos += n
    return s^


def _pickle_read_int_tuple(data: List[UInt8], mut pos: Int) raises -> List[Int]:
    """Reads an all-`Int` tuple off the pickle stream: `EMPTY_TUPLE` for
    rank 0, `<int> TUPLE1` / `<int> <int> TUPLE2` / `... TUPLE3` for ranks
    1-3, and `MARK <int>... TUPLE` for rank 4 and above -- exactly how
    CPython's pickler encodes a plain tuple of small ints, which is what
    a tensor's `size` and `stride` always are.
    """
    if pos >= len(data):
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .pt pickle stream while reading a tuple.",
                location="torch._pickle_read_int_tuple()",
            )
        )
    if data[pos] == _PICKLE_EMPTY_TUPLE:
        pos += 1
        return List[Int]()

    var used_mark = False
    if data[pos] == _PICKLE_MARK:
        used_mark = True
        pos += 1

    var out = List[Int]()
    while True:
        if pos >= len(data):
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Truncated .pt pickle stream while reading a tuple."
                    ),
                    location="torch._pickle_read_int_tuple()",
                )
            )
        var op = data[pos]
        if (
            op == _PICKLE_BININT1
            or op == _PICKLE_BININT2
            or op == _PICKLE_BININT
            or op == _PICKLE_LONG1
        ):
            out.append(_pickle_read_binint(data, pos))
            continue
        if used_mark and op == _PICKLE_TUPLE:
            pos += 1
            return out^
        if not used_mark:
            var expected = 0
            if op == _PICKLE_TUPLE1:
                expected = 1
            elif op == _PICKLE_TUPLE2:
                expected = 2
            elif op == _PICKLE_TUPLE3:
                expected = 3
            if expected > 0:
                if len(out) != expected:
                    raise Error(
                        NumojoError(
                            category="value",
                            message=(
                                "Malformed tuple in .pt pickle stream: opcode"
                                " expects {} element(s) but {} were read."
                            ).format(expected, len(out)),
                            location="torch._pickle_read_int_tuple()",
                        )
                    )
                pos += 1
                return out^
        raise Error(
            NumojoError(
                category="value",
                message="Unexpected opcode while reading a .pt pickle tuple.",
                location="torch._pickle_read_int_tuple()",
            )
        )


def _parse_rebuild_tensor_pickle(
    data: List[UInt8],
) raises -> Tuple[String, String, String, Int, Int, List[Int], List[Int], Bool]:
    """Parses a `.pt` `data.pkl` stream, returning `(storage_kind,
    storage_key, device, numel, storage_offset, shape, stride,
    requires_grad)`.

    Raises for anything other than the exact opcode shape a single
    `torch.save(tensor, path)` produces -- a state dict or a whole model
    pickles a much larger object graph and is rejected rather than
    partially read.
    """
    var pos = 0
    _pickle_expect(data, pos, _PICKLE_PROTO, "the pickle protocol marker")
    if pos >= len(data):
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .pt pickle stream.",
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )
    var proto_version = Int(data[pos])
    pos += 1
    if proto_version < 2:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Unsupported pickle protocol {} (need protocol 2 or later)."
                ).format(proto_version),
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )

    var top_call = _pickle_read_global(data, pos)
    if top_call[0] != "torch._utils" or top_call[1] != "_rebuild_tensor_v2":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "'{}.{}' does not hold a single torch.Tensor. load_torch"
                    " only reads files saved as torch.save(tensor, path)."
                ).format(top_call[0], top_call[1]),
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )
    _pickle_skip_binput(data, pos)

    _pickle_expect(data, pos, _PICKLE_MARK, "the start of the call arguments")
    _pickle_expect(
        data, pos, _PICKLE_MARK, "the start of the persistent-id tuple"
    )

    var persistent_kind = _pickle_read_binunicode(data, pos)
    if persistent_kind != "storage":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Unrecognized persistent-id kind '{}' in .pt file."
                ).format(persistent_kind),
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )
    _pickle_skip_binput(data, pos)

    var storage_class = _pickle_read_global(data, pos)
    if storage_class[0] != "torch" or not storage_class[1].endswith("Storage"):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Unrecognized storage type '{}.{}' in .pt file."
                ).format(storage_class[0], storage_class[1]),
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )
    var suffix_len = storage_class[1].byte_length() - 7  # drop "Storage"
    var storage_kind = String(storage_class[1][byte=0:suffix_len])
    _pickle_skip_binput(data, pos)

    var storage_key = _pickle_read_binunicode(data, pos)
    _pickle_skip_binput(data, pos)

    var device = _pickle_read_binunicode(data, pos)
    _pickle_skip_binput(data, pos)

    var numel = _pickle_read_binint(data, pos)

    _pickle_expect(
        data, pos, _PICKLE_TUPLE, "the end of the persistent-id tuple"
    )
    _pickle_skip_binput(data, pos)

    _pickle_expect(
        data, pos, _PICKLE_BINPERSID, "the persistent-id load of the storage"
    )

    var storage_offset = _pickle_read_binint(data, pos)
    var shape = _pickle_read_int_tuple(data, pos)
    _pickle_skip_binput(data, pos)
    var stride = _pickle_read_int_tuple(data, pos)
    _pickle_skip_binput(data, pos)

    if pos >= len(data):
        raise Error(
            NumojoError(
                category="value",
                message="Truncated .pt pickle stream.",
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )
    var requires_grad: Bool
    if data[pos] == _PICKLE_NEWTRUE:
        requires_grad = True
        pos += 1
    elif data[pos] == _PICKLE_NEWFALSE:
        requires_grad = False
        pos += 1
    else:
        raise Error(
            NumojoError(
                category="value",
                message="Expected a requires_grad flag in .pt pickle stream.",
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )

    var hooks_class = _pickle_read_global(data, pos)
    if hooks_class[0] != "collections" or hooks_class[1] != "OrderedDict":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Unrecognized backward_hooks constructor '{}.{}' in .pt"
                    " file."
                ).format(hooks_class[0], hooks_class[1]),
                location="torch._parse_rebuild_tensor_pickle()",
            )
        )
    _pickle_skip_binput(data, pos)
    _pickle_expect(
        data,
        pos,
        _PICKLE_EMPTY_TUPLE,
        "empty constructor arguments for backward_hooks",
    )
    _pickle_expect(data, pos, _PICKLE_REDUCE, "construction of backward_hooks")
    _pickle_skip_binput(data, pos)

    _pickle_expect(data, pos, _PICKLE_TUPLE, "the end of the call arguments")
    _pickle_skip_binput(data, pos)
    _pickle_expect(data, pos, _PICKLE_REDUCE, "the _rebuild_tensor_v2 call")
    _pickle_skip_binput(data, pos)
    _pickle_expect(data, pos, _PICKLE_STOP, "the end of the pickle stream")

    return (
        storage_kind^,
        storage_key^,
        device^,
        numel,
        storage_offset,
        shape^,
        stride^,
        requires_grad,
    )


def _write_binint(mut out: List[UInt8], val: Int):
    if val >= 0 and val <= 0xFF:
        out.append(_PICKLE_BININT1)
        out.append(UInt8(val))
    elif val >= 0 and val <= 0xFFFF:
        out.append(_PICKLE_BININT2)
        _put_u16(out, val)
    elif val >= -0x80000000 and val <= 0x7FFFFFFF:
        out.append(_PICKLE_BININT)
        _put_u32(out, val & 0xFFFFFFFF)
    else:
        # LONG1: a little-endian two's-complement payload, least bytes
        # first, sized to fit -- 8 bytes covers every size NuMojo can
        # allocate.
        out.append(_PICKLE_LONG1)
        out.append(UInt8(8))
        var v = val
        for _ in range(8):
            out.append(UInt8(v & 0xFF))
            v = v >> 8


def _write_int_tuple(mut out: List[UInt8], values: List[Int]):
    if len(values) == 0:
        out.append(_PICKLE_EMPTY_TUPLE)
    elif len(values) <= 3:
        for v in values:
            _write_binint(out, v)
        if len(values) == 1:
            out.append(_PICKLE_TUPLE1)
        elif len(values) == 2:
            out.append(_PICKLE_TUPLE2)
        else:
            out.append(_PICKLE_TUPLE3)
    else:
        out.append(_PICKLE_MARK)
        for v in values:
            _write_binint(out, v)
        out.append(_PICKLE_TUPLE)


def _write_binunicode(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    out.append(_PICKLE_BINUNICODE)
    _put_u32(out, len(b))
    for c in b:
        out.append(c)


def _write_global(mut out: List[UInt8], module: String, qualname: String):
    out.append(_PICKLE_GLOBAL)
    for c in module.as_bytes():
        out.append(c)
    out.append(UInt8(ord("\n")))
    for c in qualname.as_bytes():
        out.append(c)
    out.append(UInt8(ord("\n")))


def _build_rebuild_tensor_pickle(
    storage_kind: String,
    storage_key: String,
    numel: Int,
    shape: List[Int],
    stride: List[Int],
) -> List[UInt8]:
    """Builds a `data.pkl` stream for a single, `requires_grad=False`,
    `'cpu'`-device tensor -- the write-side mirror of
    `_parse_rebuild_tensor_pickle`. No `BINPUT` memo opcodes: nothing in
    this call graph is referenced twice, so they would be pure overhead,
    and `torch.load` does not require them (a real `torch.save` emits them
    only because CPython's general-purpose pickler always does).
    """
    var out = List[UInt8]()
    out.append(_PICKLE_PROTO)
    out.append(UInt8(2))
    _write_global(out, "torch._utils", "_rebuild_tensor_v2")
    out.append(_PICKLE_MARK)
    out.append(_PICKLE_MARK)
    _write_binunicode(out, "storage")
    _write_global(out, "torch", String(storage_kind, "Storage"))
    _write_binunicode(out, storage_key)
    _write_binunicode(out, "cpu")
    _write_binint(out, numel)
    out.append(_PICKLE_TUPLE)
    out.append(_PICKLE_BINPERSID)
    _write_binint(out, 0)  # storage_offset
    _write_int_tuple(out, shape)
    _write_int_tuple(out, stride)
    out.append(_PICKLE_NEWFALSE)  # requires_grad
    _write_global(out, "collections", "OrderedDict")
    out.append(_PICKLE_EMPTY_TUPLE)
    out.append(_PICKLE_REDUCE)
    out.append(_PICKLE_TUPLE)
    out.append(_PICKLE_REDUCE)
    out.append(_PICKLE_STOP)
    return out^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def load_torch[dtype: DType](path: String) raises -> NDArray[dtype]:
    """
    Reads a single tensor from a PyTorch `.pt` file, with no Python or
    `torch` involved.

    Parameters:
        dtype: Expected datatype of the NDArray elements.

    Args:
        path: File path to read from.

    Returns:
        The tensor stored in the file, as an NDArray.

    Raises:
        NumojoError: If `dtype` has no PyTorch storage equivalent; if the
            file is not an uncompressed, non-ZIP64 `.pt` archive holding a
            single plain `torch.Tensor`; if the file's dtype or device
            doesn't match what was requested; or if the tensor is 0-d.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.io.torch import load_torch

        var a = load_torch[nm.f32]("weights.pt")
        ```
    """
    comptime expected_kind = _torch_storage_kind[dtype]()
    comptime if expected_kind == "":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "{} has no PyTorch storage type, so no .pt file can hold"
                    " it."
                ).format(String(dtype)),
                location="torch.load_torch()",
            )
        )

    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()

    var pickle_data = _zip_extract(data, "data.pkl")
    var parsed = _parse_rebuild_tensor_pickle(pickle_data)
    var storage_kind = parsed[0]
    var storage_key = parsed[1]
    var device = parsed[2]
    var numel = parsed[3]
    var storage_offset = parsed[4]
    var shape = parsed[5].copy()
    var stride = parsed[6].copy()

    if device != "cpu":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "Tensor is on device '{}' -- load_torch only reads"
                    " 'cpu' tensors."
                ).format(device),
                location="torch.load_torch()",
            )
        )
    if storage_kind != expected_kind:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "dtype mismatch -- the file holds a torch.{}Storage but"
                    " {} ('{}Storage') was requested."
                ).format(storage_kind, String(dtype), expected_kind),
                location="torch.load_torch()",
            )
        )

    var raw = _zip_extract(data, String("data/", storage_key))
    var item_size = size_of[Scalar[dtype]]()
    if len(raw) != numel * item_size:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "Storage byte length doesn't match its declared element"
                    " count in the .pt file."
                ),
                location="torch.load_torch()",
            )
        )

    var ndim = len(shape)
    if ndim == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "0-d (scalar) tensors are not supported: NuMojo's"
                    " NDArray does not support rank-0 arrays yet."
                ),
                location="torch.load_torch()",
            )
        )

    var result = NDArray[dtype](NDArrayShape(shape), order="C")

    for flat in range(result.size):
        var remainder = flat
        var storage_index = storage_offset
        for d in range(ndim - 1, -1, -1):
            var extent = shape[d]
            var coord = remainder % extent
            remainder = remainder // extent
            storage_index += coord * stride[d]
        result.unsafe_set(
            flat, _read_scalar_bytes[dtype](raw, storage_index * item_size)
        )

    return result^


def save_torch[dtype: DType](path: String, A: NDArray[dtype]) raises:
    """
    Writes an NDArray to a PyTorch `.pt` file as a single, `'cpu'`,
    `requires_grad=False` tensor, with no Python or `torch` involved.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        path: File path to write to.
        A: The array to save.

    Raises:
        NumojoError: If `dtype` has no PyTorch storage equivalent (unsigned
            widths above 8 bits, `int128`, `int256`, `uint128`, `uint256`);
            or if `A` is 0-d (NuMojo's `NDArray` does not support rank-0
            arrays yet).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.io.torch import save_torch

        var a = nm.arange[nm.f32](6).reshape(nm.Shape(2, 3))
        save_torch("a.pt", a)
        ```
    """
    comptime storage_kind = _torch_storage_kind[dtype]()
    comptime if storage_kind == "":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "{} has no PyTorch storage type, so it has no .pt"
                    " representation."
                ).format(String(dtype)),
                location="torch.save_torch()",
            )
        )

    if A.ndim == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "0-d (scalar) tensors are not supported: NuMojo's"
                    " NDArray does not support rank-0 arrays yet."
                ),
                location="torch.save_torch()",
            )
        )

    var Ac = A.contiguous()

    var shape = List[Int]()
    var stride = List[Int]()
    for i in range(Ac.ndim):
        shape.append(Int(Ac.shape[i]))
        stride.append(Int(Ac.strides[i]))

    var pickle_bytes = _build_rebuild_tensor_pickle(
        storage_kind, "0", Ac.size, shape^, stride^
    )

    var raw = List[UInt8](capacity=Ac.size * size_of[Scalar[dtype]]())
    for i in range(Ac.size):
        _append_scalar_bytes[dtype](Ac.unsafe_get(i), raw)

    var byteorder_bytes = String("little").as_bytes().copy()
    var byteorder_payload = List[UInt8]()
    for c in byteorder_bytes:
        byteorder_payload.append(c)

    var version_payload = List[UInt8]()
    version_payload.append(UInt8(ord("3")))
    version_payload.append(UInt8(ord("\n")))

    var names: List[String] = [
        "archive/data.pkl",
        "archive/byteorder",
        "archive/version",
        "archive/data/0",
    ]
    var payloads: List[List[UInt8]] = [
        pickle_bytes^,
        byteorder_payload^,
        version_payload^,
        raw^,
    ]

    var zip_bytes = _zip_write_stored(names^, payloads^)
    var f = open(path, "w")
    f.write_bytes(Span(zip_bytes))
    f.close()
