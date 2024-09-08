from python import Python, PythonObject
from testing.testing import assert_true
import numojo as nm


fn check[dtype: DType](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


fn check_is_close[dtype: DType](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.isclose(array.to_numpy(), np_sol)), st)
