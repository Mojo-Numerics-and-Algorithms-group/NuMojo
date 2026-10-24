from std.python import Python, PythonObject

from numojo.prelude import *
from std.testing.testing import (
    assert_true,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from utils_for_test import check, check_is_close, check_values_close
from std.testing import TestSuite


def test_constructors() raises:
    # Test NDArray constructor with different input types
    var arr1 = NDArray[f32](Shape(3, 4, 5))
    assert_true(arr1.ndim == 3, "NDArray constructor: ndim")
    assert_true(arr1.shape[0] == 3, "NDArray constructor: shape element 0")
    assert_true(arr1.shape[1] == 4, "NDArray constructor: shape element 1")
    assert_true(arr1.shape[2] == 5, "NDArray constructor: shape element 2")
    assert_true(arr1.size == 60, "NDArray constructor: size")
    assert_true(arr1.dtype == DType.float32, "NDArray constructor: dtype")

    var arr3 = nm.full[f32](Shape(3, 4, 5), fill_value=Scalar[f32](10.0))
    # maybe it's better to return a scalar for arr[0, 0, 0]
    assert_equal(
        arr3[Item(0, 0, 0)], 10.0, "NDArray constructor with fill value"
    )

    var values: List[Int] = [3, 4, 5]
    var arr4 = NDArray[f32](NDArrayShape(values))
    assert_true(
        arr4.shape[0] == 3, "NDArray constructor with List: shape element 0"
    )
    assert_true(
        arr4.shape[1] == 4, "NDArray constructor with List: shape element 1"
    )
    assert_true(
        arr4.shape[2] == 5, "NDArray constructor with List: shape element 2"
    )

    var arr5 = NDArray[f32](NDArrayShape(3, 4, 5))
    assert_true(
        arr5.shape[0] == 3,
        "NDArray constructor with NDArrayShape: shape element 0",
    )
    assert_true(
        arr5.shape[1] == 4,
        "NDArray constructor with NDArrayShape: shape element 1",
    )
    assert_true(
        arr5.shape[2] == 5,
        "NDArray constructor with NDArrayShape: shape element 2",
    )


def test_iterator() raises:
    var py = Python.import_module("builtins")
    var np = Python.import_module("numpy")

    var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var anp = a.to_numpy()
    var f = nm.arange[i8](24).reshape(Shape(2, 3, 4), order="F")
    var fnp = f.to_numpy()

    # NDAxisIter
    var a_iter_along_axis = a.iter_along_axis[forward=False](axis=0)
    var b = a_iter_along_axis.__next__() == nm.array[i8]("[11, 23]")
    assert_true(
        b.item(0) == True,
        "`_NDAxisIter` breaks",
    )
    assert_true(
        b.item(1) == True,
        "`_NDAxisIter` breaks",
    )

    # NDArrayIter
    var a_iter_over_dimension = a.__iter__()
    var anp_iter_over_dimension = anp.__iter__()
    for _ in range(a.shape[0]):
        check(
            a_iter_over_dimension.__next__(),
            anp_iter_over_dimension.__next__(),
            "`_NDArrayIter` or `__iter__()` breaks",
        )

    var a_iter_over_dimension_reversed = a.__reversed__()
    var anp_iter_over_dimension_reversed = py.reversed(anp)
    for _ in range(a.shape[0]):
        check(
            a_iter_over_dimension_reversed.__next__(),
            anp_iter_over_dimension_reversed.__next__(),
            "`_NDArrayIter` or `__reversed__()` breaks",
        )

    # NDIter of C-order array
    var a_nditer = a.nditer()
    var anp_nditer = np.nditer(anp)
    for _ in range(a.size):
        var a_val = a_nditer.__next__()
        var anp_val = anp_nditer.__next__()
        check_values_close(
            a_val,
            anp_val,
            "`_NDIter` or `nditer()` of C array by order C breaks",
        )

    # NDIter of C-order array
    a_nditer = a.nditer()
    anp_nditer = np.nditer(anp)
    for i in range(a.size):
        check_values_close(
            a_nditer.ith(i),
            anp_nditer.__next__(),
            "`_NDIter.ith()` of C array by order C breaks",
        )

    var a_nditer_f = a.nditer(order="F")
    var anp_nditer_f = np.nditer(anp, order=PythonObject("F"))
    for _ in range(a.size):
        var a_val = a_nditer_f.__next__()
        var anp_val = anp_nditer_f.__next__()
        check_values_close(
            a_val,
            anp_val,
            "`_NDIter` or `nditer()` of C array by order F breaks",
        )

    # NDIter of F-order array
    var f_nditer = f.nditer()
    var fnp_nditer = np.nditer(fnp)
    for _ in range(f.size):
        var f_val = f_nditer.__next__()
        var fnp_val = fnp_nditer.__next__()
        check_values_close(
            f_val,
            fnp_val,
            "`_NDIter` or `nditer()` of F array by order C breaks",
        )

    var f_nditer_f = f.nditer(order="F")
    var fnp_nditer_f = np.nditer(fnp, order=PythonObject("F"))
    for _ in range(f.size):
        var f_val = f_nditer_f.__next__()
        var fnp_val = fnp_nditer_f.__next__()
        check_values_close(
            f_val,
            fnp_val,
            "`_NDIter` or `nditer()` of F array by order F breaks",
        )


def test_swapaxes_method() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.rand(2, 3, 4)
    check(
        a.swapaxes(0, 2),
        np.swapaxes(a.to_numpy(), 0, 2),
        "`NDArray.swapaxes()` is broken",
    )


def test_repeat_method() raises:
    var np = Python.import_module("numpy")
    var a = nm.fromstring[nm.i32]("[1, 2, 3]")
    check(
        a.repeat(2), np.repeat(a.to_numpy(), 2), "`NDArray.repeat()` is broken"
    )

    var b = nm.reshape(nm.fromstring[nm.i32]("[1, 2, 3, 4]"), Shape(2, 2))
    check(
        b.repeat(2, axis=0),
        np.repeat(b.to_numpy(), 2, axis=0),
        "`NDArray.repeat()` with axis is broken",
    )


def test_partition_argpartition_methods() raises:
    var a = nm.fromstring[nm.i32]("[3, 1, 4, 1, 5, 9, 2, 6]")
    var sorted_a = nm.sort(a)
    var kth = 3

    var part = a.partition(kth)
    assert_true(
        part.item(kth) == sorted_a.item(kth),
        "`NDArray.partition()` kth value mismatch",
    )

    var indices = a.argpartition(kth)
    var reordered = NDArray[nm.i32](Shape(a.size))
    for i in range(a.size):
        reordered[Item(i)] = a.item(Int(indices.item(i)))
    assert_true(
        reordered.item(kth) == sorted_a.item(kth),
        "`NDArray.argpartition()` kth value mismatch",
    )


def test_choose() raises:
    var np = Python.import_module("numpy")
    var index = nm.fromstring[nm.i32]("[0, 1, 0, 1]")
    var c0 = nm.fromstring[nm.i32]("[10, 20, 30, 40]")
    var c1 = nm.fromstring[nm.i32]("[1, 2, 3, 4]")
    var choices = List[NDArray[nm.i32]]()
    choices.append(c0.copy())
    choices.append(c1.copy())

    var index_np = index.to_numpy()
    var choices_np = Python.list(c0.to_numpy(), c1.to_numpy())
    check(
        index.choose(choices),
        index_np.choose(choices_np),
        "`NDArray.choose()` is broken",
    )

    with assert_raises():
        var bad_index = nm.fromstring[nm.i32]("[0, 5]")
        _ = bad_index.choose(choices)


def test_byteswap() raises:
    var np = Python.import_module("numpy")
    var a = nm.fromstring[nm.i32]("[1, 256, 65536]")
    var anp = a.to_numpy()
    check(a.byteswap(), anp.byteswap(), "`NDArray.byteswap()` is broken")

    var b = nm.fromstring[nm.i32]("[1, 256, 65536]")
    var original = b.copy()
    _ = b.byteswap(inplace=True)
    assert_true(
        b.item(0) != original.item(0),
        "`NDArray.byteswap(inplace=True)` did not mutate the array",
    )


def test_getfield_setfield() raises:
    var np = Python.import_module("numpy")
    var a = nm.fromstring[nm.f64]("[1.0, 2.0]")
    var anp = a.to_numpy()
    check(
        a.getfield[nm.i64](),
        anp.getfield(np.int64),
        "`NDArray.getfield()` is broken",
    )

    var zeros_bits = nm.fromstring[nm.i64]("[0, 0]")
    var b = a.copy()
    b.setfield(zeros_bits)
    assert_true(
        b.item(0) == 0.0, "`NDArray.setfield()` did not zero the bit pattern"
    )


def test_tobytes_tofile_dump() raises:
    var np = Python.import_module("numpy")
    var pickle = Python.import_module("pickle")
    var builtins = Python.import_module("builtins")

    var a = nm.fromstring[nm.i32]("[1, 2, 3]")
    var raw = a.tobytes()
    assert_true(
        Bool(raw == a.to_numpy().tobytes()), "`NDArray.tobytes()` is broken"
    )

    var tofile_path = "/tmp/numojo_test_tofile.bin"
    a.tofile(tofile_path)
    var reloaded = np.fromfile(tofile_path, dtype=np.int32)
    check(a, reloaded, "`NDArray.tofile()` is broken")

    var dump_path = "/tmp/numojo_test_dump.pkl"
    a.dump(dump_path)
    var f = builtins.open(dump_path, "rb")
    var loaded = pickle.load(f)
    f.close()
    check(a, loaded, "`NDArray.dump()` is broken")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
