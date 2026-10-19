from std.python import Python, PythonObject
from std.testing import TestSuite
from std.testing.testing import assert_true
from utils_for_test import check, check_is_close

import numojo as nm
from numojo.prelude import *


def check_float_result[
    dtype: DType
](actual: nm.NDArray[dtype], expected: PythonObject, message: String) raises:
    var np = Python.import_module("numpy")
    var actual_np = actual.to_numpy()
    assert_true(np.array_equal(actual_np, expected, equal_nan=True), message)
    assert_true(
        np.array_equal(np.signbit(actual_np), np.signbit(expected)),
        message + " signbit",
    )


def test_unary_elementwise_against_numpy() raises:
    var np = Python.import_module("numpy")
    var values = nm.array[nm.f64]("[-3.5, -2.5, -0.0, 0.0, 1.5, 2.5]")
    var values_np = values.to_numpy()

    check_float_result(nm.sign(values), np.sign(values_np), "sign")
    check_float_result(nm.square(values), np.square(values_np), "square")
    check_float_result(nm.negative(values), np.negative(values_np), "negative")
    check_float_result(nm.positive(values), np.positive(values_np), "positive")
    check_float_result(nm.rint(values), np.rint(values_np), "rint")


def test_power_against_numpy() raises:
    var np = Python.import_module("numpy")
    var base = nm.array[nm.f64]("[1.0, 2.0, 3.0, 4.0]")
    var exponent = nm.array[nm.f64]("[4.0, 3.0, 2.0, 0.5]")
    var scalar: Scalar[nm.f64] = 2.0

    check_is_close(
        nm.power(base, exponent),
        np.power(base.to_numpy(), exponent.to_numpy()),
        "power array-array",
    )
    check_is_close(
        nm.pow(base, scalar),
        np.pow(base.to_numpy(), 2.0),
        "pow array-scalar alias",
    )
    check_is_close(
        nm.power(scalar, exponent),
        np.power(2.0, exponent.to_numpy()),
        "power scalar-array",
    )

    var int_base = nm.array[nm.i32]("[1, 2, 3, 4]")
    var int_exponent = nm.array[nm.i32]("[4, 3, 2, 1]")
    check(
        nm.power(int_base, int_exponent),
        np.power(int_base.to_numpy(), int_exponent.to_numpy()),
        "integer power",
    )


def test_fmax_fmin_against_numpy() raises:
    var np = Python.import_module("numpy")
    var nan64 = Python.float("nan").__float__()
    var lhs_np = np.array(
        Python.list(nan64, 4.0, -0.0, 0.0, nan64, -3.0),
        dtype=np.float64,
    )
    var rhs_np = np.array(
        Python.list(2.0, nan64, 0.0, -0.0, nan64, -5.0),
        dtype=np.float64,
    )
    var lhs = nm.array[nm.f64](lhs_np)
    var rhs = nm.array[nm.f64](rhs_np)
    var scalar: Scalar[nm.f64] = 1.5

    check_float_result(
        nm.fmax(lhs, rhs), np.fmax(lhs_np, rhs_np), "fmax array-array"
    )
    check_float_result(
        nm.fmin(lhs, rhs), np.fmin(lhs_np, rhs_np), "fmin array-array"
    )
    check_float_result(
        nm.fmax(lhs, scalar), np.fmax(lhs_np, 1.5), "fmax array-scalar"
    )
    check_float_result(
        nm.fmin(scalar, rhs), np.fmin(1.5, rhs_np), "fmin scalar-array"
    )


def test_fmod_against_numpy() raises:
    var np = Python.import_module("numpy")
    var lhs = nm.array[nm.f64]("[-7.5, -5.0, 5.0, 7.5]")
    var rhs = nm.array[nm.f64]("[2.0, -2.0, 2.0, -2.0]")
    var scalar: Scalar[nm.f64] = -2.0

    check_float_result(
        nm.fmod(lhs, rhs),
        np.fmod(lhs.to_numpy(), rhs.to_numpy()),
        "fmod array-array",
    )
    check_float_result(
        nm.fmod(lhs, scalar),
        np.fmod(lhs.to_numpy(), -2.0),
        "fmod array-scalar",
    )
    check_float_result(
        nm.fmod(scalar, rhs),
        np.fmod(-2.0, rhs.to_numpy()),
        "fmod scalar-array",
    )

    var int_lhs = nm.array[nm.i32]("[-7, -5, 5, 7]")
    var int_rhs = nm.array[nm.i32]("[2, -2, 2, -2]")
    check(
        nm.fmod(int_lhs, int_rhs),
        np.fmod(int_lhs.to_numpy(), int_rhs.to_numpy()),
        "integer fmod",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
