from std.python import Python
from std import os
from std.testing import TestSuite

import numojo as nm
from numojo.prelude import *
from numojo.routines.io.files import load, save, loadtxt, savetxt
from numojo.routines.io.npy import load_npy, save_npy
from numojo import ones, full
from utils_for_test import check_is_close


def test_save_and_load() raises:
    var np = Python.import_module("numpy")
    var arr = ones[nm.f32](nm.Shape(10, 15))
    var fname = "test_save_load.npy"
    save(fname=fname, array=arr)
    # Load with numpy for cross-check
    var np_loaded = np.load(fname)
    np.allclose(np_loaded, arr.to_numpy())
    # Load with numojo
    var arr2 = load(fname)
    np.allclose(arr2.to_numpy(), arr.to_numpy())
    # Clean up
    os.remove(fname)


def test_savetxt_and_loadtxt() raises:
    var np = Python.import_module("numpy")
    var arr = full[nm.f32](nm.Shape(10, 15), fill_value=5.0)
    var fname = "test_savetxt_loadtxt.txt"
    savetxt(fname, arr, fmt="%.2f")
    # Load with numpy for cross-check
    var np_loaded = np.loadtxt(fname)
    np.allclose(np_loaded, arr.to_numpy())
    # Load with numojo
    var arr2 = loadtxt(fname)
    np.allclose(arr2.to_numpy(), arr.to_numpy())
    # Clean up
    os.remove(fname)


def test_save_npy_and_load_npy() raises:
    var np = Python.import_module("numpy")
    var arr = nm.random.randn(3, 4)

    var fname = "test_npy_roundtrip.npy"
    save_npy(fname, arr)

    # numpy can read what our pure-Mojo writer produced.
    var np_loaded = np.load(fname)
    check_is_close(arr, np_loaded, "`save_npy` output unreadable by numpy.")

    # Mojo reader round trip.
    var arr2 = load_npy[nm.f64](fname)
    check_is_close(arr2, arr.to_numpy(), "`load_npy` roundtrip is broken.")
    os.remove(fname)


def test_load_npy_reads_numpy_file() raises:
    var np = Python.import_module("numpy")
    var np_arr = np.arange(24, dtype=np.int32).reshape(Python.tuple(2, 3, 4))

    var fname = "test_npy_from_numpy.npy"
    np.save(fname, np_arr)
    var arr = load_npy[nm.i32](fname)
    check_is_close(arr, np_arr, "`load_npy` cannot read a numpy-written file.")
    os.remove(fname)


def test_load_npy_fortran_order() raises:
    var np = Python.import_module("numpy")
    var np_arr = np.asfortranarray(
        np.arange(12, dtype=np.float32).reshape(Python.tuple(3, 4))
    )

    var fname = "test_npy_fortran.npy"
    np.save(fname, np_arr)
    var arr = load_npy[nm.f32](fname)
    check_is_close(
        arr, np_arr, "`load_npy` cannot read a Fortran-order numpy file."
    )
    os.remove(fname)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
