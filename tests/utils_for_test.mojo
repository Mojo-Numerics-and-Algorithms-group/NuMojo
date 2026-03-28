from std.python import Python, PythonObject
from std.testing.testing import assert_true
import numojo as nm


def check[
    dtype: DType, //
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


def check_with_dtype[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(String(array.dtype) == String(np_sol.dtype), "DType mismatch")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


def check_is_close[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(array.to_numpy(), np_sol, atol=PythonObject(0.1))), st
    )


def check_values_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=PythonObject(0.001)), st)
