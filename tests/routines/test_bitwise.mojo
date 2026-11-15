from std.python import Python, PythonObject
from std.testing import TestSuite

import numojo as nm
from numojo.prelude import *
from utils_for_test import check_with_dtype


def test_bitwise_binary_operations() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.i32]("[[0, 1, 2], [3, 7, 15]]")
    var b = nm.array[nm.i32]("[[1, 3, 6], [5, 10, 12]]")
    var a_np = a.to_numpy()
    var b_np = b.to_numpy()

    check_with_dtype(
        nm.bitwise_and(a, b),
        np.bitwise_and(a_np, b_np),
        "bitwise_and must match NumPy",
    )
    check_with_dtype(
        nm.bitwise_or(a, b),
        np.bitwise_or(a_np, b_np),
        "bitwise_or must match NumPy",
    )
    check_with_dtype(
        nm.bitwise_xor(a, b),
        np.bitwise_xor(a_np, b_np),
        "bitwise_xor must match NumPy",
    )


def test_bitwise_binary_broadcasting() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.i32]("[[1, 2, 3], [4, 5, 6]]")
    var b = nm.array[nm.i32]("[1, 3, 7]")

    check_with_dtype(
        nm.bitwise_and(a, b),
        np.bitwise_and(a.to_numpy(), b.to_numpy()),
        "bitwise_and broadcasting must match NumPy",
    )
    check_with_dtype(
        nm.bitwise_or(a, b),
        np.bitwise_or(a.to_numpy(), b.to_numpy()),
        "bitwise_or broadcasting must match NumPy",
    )
    check_with_dtype(
        nm.bitwise_xor(a, b),
        np.bitwise_xor(a.to_numpy(), b.to_numpy()),
        "bitwise_xor broadcasting must match NumPy",
    )


def test_bit_shifts() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.i32]("[1, 2, 4, 8, 16]")
    var a_np = a.to_numpy()

    check_with_dtype(
        nm.left_shift(a, 2),
        np.left_shift(a_np, PythonObject(2)),
        "left_shift must match NumPy",
    )
    check_with_dtype(
        nm.right_shift(a, 2),
        np.right_shift(a_np, PythonObject(2)),
        "right_shift must match NumPy",
    )


def test_invert() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.i32]("[-8, -1, 0, 1, 7]")

    check_with_dtype(
        nm.invert(a),
        np.invert(a.to_numpy()),
        "invert must match NumPy",
    )


def test_bitwise_count() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.i32]("[0, 1, 2, 3, 7, 255, 256]")

    check_with_dtype(
        nm.bitwise_count(a),
        np.bitwise_count(a.to_numpy()),
        "bitwise_count must match NumPy",
    )


def test_packbits_unpackbits_flat() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.boolean]("[1, 0, 1, 1, 0, 0, 0, 0, 1]")
    var a_np = a.to_numpy()

    var packed = nm.packbits(a)
    check_with_dtype(
        packed, np.packbits(a_np), "packbits (flat) must match NumPy"
    )

    check_with_dtype(
        nm.unpackbits(packed),
        np.unpackbits(np.packbits(a_np)),
        "unpackbits (flat) must match NumPy",
    )


def test_packbits_unpackbits_axis() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.u8](
        "[[1, 0, 1, 1, 0], [0, 0, 1, 0, 1], [1, 1, 1, 1, 1]]"
    )
    var a_np = a.to_numpy()

    check_with_dtype(
        nm.packbits(a, axis=1),
        np.packbits(a_np, axis=1),
        "packbits (axis=1) must match NumPy",
    )
    check_with_dtype(
        nm.packbits(a, axis=0),
        np.packbits(a_np, axis=0),
        "packbits (axis=0) must match NumPy",
    )

    var packed = nm.packbits(a, axis=1)
    var packed_np = np.packbits(a_np, axis=1)
    check_with_dtype(
        nm.unpackbits(packed, axis=1, count=5),
        np.unpackbits(packed_np, axis=1, count=5),
        "unpackbits (axis=1, count) must match NumPy",
    )


def test_packbits_little_endian() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.boolean]("[1, 0, 1, 1, 0]")
    var a_np = a.to_numpy()

    check_with_dtype(
        nm.packbits(a, bitorder="little"),
        np.packbits(a_np, bitorder="little"),
        "packbits (little-endian) must match NumPy",
    )
    var packed = nm.packbits(a, bitorder="little")
    check_with_dtype(
        nm.unpackbits(packed, bitorder="little"),
        np.unpackbits(np.packbits(a_np, bitorder="little"), bitorder="little"),
        "unpackbits (little-endian) must match NumPy",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
