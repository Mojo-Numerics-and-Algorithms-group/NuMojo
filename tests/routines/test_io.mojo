from numojo.routines.io.files import load, save, loadtxt, savetxt
from numojo import ones, full
from python import Python
import os


fn test_save_and_load() raises:
    var np = Python.import_module("numpy")
    var arr = ones[numojo.f32](numojo.Shape(10, 15))
    var fname = "test_save_load.npy"
    save(fname, arr)
    # Load with numpy for cross-check
    var np_loaded = np.load(fname)
    np.allclose(np_loaded, arr.to_numpy())
    # Load with numojo
    var arr2 = load(fname)
    np.allclose(arr2.to_numpy(), arr.to_numpy())
    # Clean up
    os.remove(fname)


fn test_savetxt_and_loadtxt() raises:
    var np = Python.import_module("numpy")
    var arr = full[numojo.f32](numojo.Shape(10, 15), fill_value=5.0)
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
