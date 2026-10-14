from std.python import Python, PythonObject
from std.testing.testing import assert_raises, assert_true
from utils_for_test import (
    check,
    check_is_close,
    check_values_close,
    check_with_dtype,
)
from std.testing import TestSuite

import numojo as nm
from numojo.prelude import *

# ===-----------------------------------------------------------------------===#
# Sums, products, differences
# ===-----------------------------------------------------------------------===#


def test_sum_prod() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2, 3, 4)
    var Anp = A.to_numpy()

    check_values_close(
        nm.sum(A),
        np.sum(Anp, axis=PythonObject(None)),
        String("`sum` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("`sum` by axis {} fails.").format(i),
        )

    check_is_close(
        nm.cumsum(A),
        np.cumsum(Anp, axis=PythonObject(None)),
        String("`cumsum` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.cumsum(A, axis=i),
            np.cumsum(Anp, axis=i),
            String("`cumsum` by axis {} fails.").format(i),
        )

    check_values_close(
        nm.prod(A),
        np.prod(Anp, axis=PythonObject(None)),
        String("`prod` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.prod(A, axis=i),
            np.prod(Anp, axis=i),
            String("`prod` by axis {} fails.").format(i),
        )

    check_is_close(
        nm.cumprod(A),
        np.cumprod(Anp, axis=PythonObject(None)),
        String("`cumprod` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.cumprod(A, axis=i),
            np.cumprod(Anp, axis=i),
            String("`cumprod` by axis {} fails.").format(i),
        )


def test_add_array() raises:
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check(nm.add[nm.f64](arr, 5.0), np.arange(0, 15) + 5, "Add array + scalar")
    check(
        nm.add[nm.f64](arr, arr),
        np.arange(0, 15) + np.arange(0, 15),
        "Add array + array",
    )


# def test_dunder_add_array() raises:
#     var np = Python.import_module("numpy")

#     # Test float + float
#     var arr_f64 = nm.arange[nm.f64](0, 15)
#     var arr_f32 = nm.arange[nm.f32](0, 15)
#     var scalarf64: Scalar[nm.f64] = 5.0
#     var scalarf32: Scalar[nm.f32] = 5.0
#     check_with_dtype(
#         arr_f64 + scalarf64,
#         np.arange(0, 15, dtype=np.float64) + 5.0,
#         "Add f64 array + f64 scalar",
#     )
#     check_with_dtype(
#         arr_f32 + scalarf32,
#         np.arange(0, 15, dtype=np.float32) + 5.0,
#         "Add f32 array + f32 scalar",
#     )
#     check_with_dtype(
#         arr_f64 + arr_f64,
#         np.arange(0, 15, dtype=np.float64) + np.arange(0, 15, dtype=np.float64),
#         "Add f64 array + f64 array",
#     )

#     # Test float + int
#     var arr_i32 = nm.arange[nm.i32](0, 15)
#     check_with_dtype(
#         arr_f64 + 5,
#         np.arange(0, 15, dtype=np.float64) + 5,
#         "Add f64 array + int scalar",
#     )
#     check_with_dtype(
#         arr_f64 + arr_i32,
#         np.arange(0, 15, dtype=np.float64) + np.arange(0, 15, dtype=np.int32),
#         "Add f64 array + i32 array",
#     )

#     # Test uint + int
#     var arr_u32 = nm.arange[nm.u32](0, 15)
#     check_with_dtype(
#         arr_u32 + 5,
#         np.arange(0, 15, dtype=np.uint32) + 5,
#         "Add u32 array + int scalar",
#     )
#     check_with_dtype(
#         arr_u32 + arr_i32,
#         np.arange(0, 15, dtype=np.uint32) + np.arange(0, 15, dtype=np.int32),
#         "Add u32 array + i32 array",
#     )


# def test_dunder_sub_array() raises:
#     var np = Python.import_module("numpy")

#     # Test float - float
#     var arr_f64 = nm.arange[nm.f64](0, 15)
#     var arr_f32 = nm.arange[nm.f32](0, 15)
#     var scalarf64: Scalar[nm.f64] = 5.0
#     var scalarf32: Scalar[nm.f32] = 5.0
#     check_with_dtype(
#         arr_f64 - scalarf64,
#         np.arange(0, 15, dtype=np.float64) - 5.0,
#         "Sub f64 array - f64 scalar",
#     )
#     check_with_dtype(
#         arr_f32 - scalarf32,
#         np.arange(0, 15, dtype=np.float32) - 5.0,
#         "Sub f32 array - f32 scalar",
#     )
#     check_with_dtype(
#         arr_f64 - arr_f64,
#         np.arange(0, 15, dtype=np.float64) - np.arange(0, 15, dtype=np.float64),
#         "Sub f64 array - f64 array",
#     )

#     # Test float - int
#     var arr_i32 = nm.arange[nm.i32](0, 15)
#     check_with_dtype(
#         arr_f64 - 5,
#         np.arange(0, 15, dtype=np.float64) - 5,
#         "Sub f64 array - int scalar",
#     )
#     check_with_dtype(
#         arr_f64 - arr_i32,
#         np.arange(0, 15, dtype=np.float64) - np.arange(0, 15, dtype=np.int32),
#         "Sub f64 array - i32 array",
#     )

#     # Test uint - int
#     var arr_u32 = nm.arange[nm.u32](0, 15)
#     check_with_dtype(
#         arr_u32 - 5,
#         np.arange(0, 15, dtype=np.uint32) - 5,
#         "Sub u32 array - int scalar",
#     )
#     check_with_dtype(
#         arr_u32 - arr_i32,
#         np.arange(0, 15, dtype=np.uint32) - np.arange(0, 15, dtype=np.int32),
#         "Sub u32 array - i32 array",
#     )


# def test_dunder_mul_array():
#     var np = Python.import_module("numpy")

#     # Test float * float
#     var arr_f64 = nm.arange[nm.f64](0, 15)
#     var arr_f32 = nm.arange[nm.f32](0, 15)
#     var scalarf64: Scalar[nm.f64] = 5.0
#     var scalarf32: Scalar[nm.f32] = 5.0
#     check_with_dtype(
#         arr_f64 * scalarf64,
#         np.arange(0, 15, dtype=np.float64) * 5.0,
#         "Mul f64 array * f64 scalar",
#     )
#     check_with_dtype(
#         arr_f32 * scalarf32,
#         np.arange(0, 15, dtype=np.float32) * 5.0,
#         "Mul f32 array * f32 scalar",
#     )
#     check_with_dtype(
#         arr_f64 * arr_f64,
#         np.arange(0, 15, dtype=np.float64) * np.arange(0, 15, dtype=np.float64),
#         "Mul f64 array * f64 array",
#     )

#     # Test float * int
#     var arr_i32 = nm.arange[nm.i32](0, 15)
#     check_with_dtype(
#         arr_f64 * 5,
#         np.arange(0, 15, dtype=np.float64) * 5,
#         "Mul f64 array * int scalar",
#     )
#     check_with_dtype(
#         arr_f64 * arr_i32,
#         np.arange(0, 15, dtype=np.float64) * np.arange(0, 15, dtype=np.int32),
#         "Mul f64 array * i32 array",
#     )

#     # Test uint * int
#     var arr_u32 = nm.arange[nm.u32](0, 15)
#     check_with_dtype(
#         arr_u32 * 5,
#         np.arange(0, 15, dtype=np.uint32) * 5,
#         "Mul u32 array * int scalar",
#     )
#     check_with_dtype(
#         arr_u32 * arr_i32,
#         np.arange(0, 15, dtype=np.uint32) * np.arange(0, 15, dtype=np.int32),
#         "Mul u32 array * i32 array",
#     )


# def test_dunder_div_array():
#     var np = Python.import_module("numpy")

#     # Test float / float
#     var arr_f64 = nm.arange[nm.f64](
#         1, 16
#     )  # Start from 1 to avoid division by 0
#     var arr_f32 = nm.arange[nm.f32](1, 16)
#     var scalarf64: Scalar[nm.f64] = 5.0
#     var scalarf32: Scalar[nm.f32] = 5.0
#     check_with_dtype(
#         arr_f64 / scalarf64,
#         np.arange(1, 16, dtype=np.float64) / 5.0,
#         "Div f64 array / f64 scalar",
#     )
#     check_with_dtype(
#         arr_f32 / scalarf32,
#         np.arange(1, 16, dtype=np.float32) / 5.0,
#         "Div f32 array / f32 scalar",
#     )
#     check_with_dtype(
#         arr_f64 / arr_f64,
#         np.arange(1, 16, dtype=np.float64) / np.arange(1, 16, dtype=np.float64),
#         "Div f64 array / f64 array",
#     )

#     # Test float / int
#     var arr_i32 = nm.arange[nm.i32](1, 16)
#     check_with_dtype(
#         arr_f64 / 5,
#         np.arange(1, 16, dtype=np.float64) / 5,
#         "Div f64 array / int scalar",
#     )
#     check_with_dtype(
#         arr_f64 / arr_i32,
#         np.arange(1, 16, dtype=np.float64) / np.arange(1, 16, dtype=np.int32),
#         "Div f64 array / i32 array",
#     )


# def test_dunder_floordiv_array():
#     var np = Python.import_module("numpy")

#     # Test float // float
#     var arr_f64 = nm.arange[nm.f64](1, 16)
#     var arr_f32 = nm.arange[nm.f32](1, 16)
#     var scalarf64: Scalar[nm.f64] = 5.0
#     var scalarf32: Scalar[nm.f32] = 5.0
#     check_with_dtype(
#         arr_f64 // scalarf64,
#         np.arange(1, 16, dtype=np.float64) // 5.0,
#         "Floordiv f64 array // f64 scalar",
#     )
#     check_with_dtype(
#         arr_f32 // scalarf32,
#         np.arange(1, 16, dtype=np.float32) // 5.0,
#         "Floordiv f32 array // f32 scalar",
#     )
#     check_with_dtype(
#         arr_f64 // arr_f64,
#         np.arange(1, 16, dtype=np.float64)
#         // np.arange(1, 16, dtype=np.float64),
#         "Floordiv f64 array // f64 array",
#     )

#     # Test float // int
#     var arr_i32 = nm.arange[nm.i32](1, 16)
#     check_with_dtype(
#         arr_f64 // 5,
#         np.arange(1, 16, dtype=np.float64) // 5,
#         "Floordiv f64 array // int scalar",
#     )
#     check_with_dtype(
#         arr_f64 // arr_i32,
#         np.arange(1, 16, dtype=np.float64) // np.arange(1, 16, dtype=np.int32),
#         "Floordiv f64 array // i32 array",
#     )


# def test_dunder_mod_array():
#     var np = Python.import_module("numpy")

#     # Test float % float
#     var arr_f64 = nm.arange[nm.f64](1, 15)
#     var arr_f32 = nm.arange[nm.f32](1, 15)
#     var arr_f321 = nm.arange[nm.f32](1, 15)
#     var scalarf64: Scalar[nm.f64] = 5.0
#     var scalarf32: Scalar[nm.f32] = 5.0
#     check_with_dtype(
#         arr_f64 % scalarf64,
#         np.arange(1, 15, dtype=np.float64) % 5.0,
#         "Mod f64 array % f64 scalar",
#     )
#     check_with_dtype(
#         arr_f32 % scalarf32,
#         np.arange(1, 15, dtype=np.float32) % 5.0,
#         "Mod f32 array % f32 scalar",
#     )
#     check_with_dtype(
#         arr_f321 % arr_f32,
#         np.arange(1, 15, dtype=np.float32) % np.arange(1, 15, dtype=np.float32),
#         "Mod f32 array % f32 array",
#     )

#     # Test float % int
#     var arr_i32 = nm.arange[nm.i32](1, 15)
#     var arr_i321 = nm.arange[nm.i32](1, 15)
#     check_with_dtype(
#         arr_f64 % 5,
#         np.arange(1, 15, dtype=np.float64) % 5,
#         "Mod f64 array % int scalar",
#     )
#     check_with_dtype(
#         arr_i321 % arr_i32,
#         np.arange(1, 15, dtype=np.int32) % np.arange(1, 15, dtype=np.int32),
#         "Mod f64 array % i32 array",
#     )


def test_add_array_par() raises:
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 20)

    check(
        nm.add[nm.f64](arr, 5.0),
        np.arange(0, 20) + 5,
        "Add array + scalar",
    )
    check(
        nm.add[nm.f64](arr, arr),
        np.arange(0, 20) + np.arange(0, 20),
        "Add array + array",
    )


def test_sin() raises:
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[nm.f64](arr), np.sin(np.arange(0, 15)), "Add array + scalar"
    )


def test_extrema() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(10)
    var anp = a.to_numpy()
    var c = nm.random.randn(2, 3, 4)
    var cnp = c.to_numpy()
    var cf = c.reshape(c.shape, order="F")  # 3d array F order
    var cfnp = cf.to_numpy()

    # max
    check_values_close(
        nm.max(a), np.max(anp, axis=PythonObject(None)), "`sort` 1d is broken"
    )
    for i in range(3):
        check(
            nm.max(c, axis=i),
            np.max(cnp, axis=i),
            String("`sort` 3d c-order by axis {} is broken").format(i),
        )
        check(
            nm.max(cf, axis=i),
            np.max(cfnp, axis=i),
            String("`sort` 3d f-order by axis {} is broken").format(i),
        )


def test_misc() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(10)
    var anp = a.to_numpy()
    var c = nm.random.randn(2, 3, 4)
    var cnp = c.to_numpy()
    var cf = c.reshape(c.shape, order="F")  # 3d array F order
    var cfnp = cf.to_numpy()

    # clip
    check(
        nm.clip(a, -0.02, 0.3),
        np.clip(anp, -0.02, 0.3),
        String("`clip` 1d c-order is broken"),
    )
    check(
        nm.clip(c, -0.02, 0.3),
        np.clip(cnp, -0.02, 0.3),
        String("`clip` 3d c-order is broken"),
    )
    check(
        nm.clip(cf, 0.02, -0.01),
        np.clip(cfnp, 0.02, -0.01),
        String("`clip` 3d f-order is broken"),
    )


def test_integrate() raises:
    var np = Python.import_module("numpy")
    var sp_integrate = Python.import_module("scipy.integrate")

    var y = nm.random.randn(10)
    var ynp = y.to_numpy()
    var x = nm.fromstring("[0.0, 1.0, 2.5, 3.0, 5.0, 5.5, 7.0, 8.0, 8.5, 10.0]")
    var xnp = x.to_numpy()

    check_values_close(
        nm.trapezoid(y), np.trapezoid(ynp), "`trapezoid` is broken"
    )
    check_values_close(
        nm.trapezoid(y, x),
        np.trapezoid(ynp, xnp),
        "`trapezoid` with x is broken",
    )

    check(
        nm.cumtrapz(y),
        sp_integrate.cumulative_trapezoid(ynp),
        "`cumtrapz` is broken",
    )
    check(
        nm.cumtrapz(y, initial=Float64(0.0)),
        sp_integrate.cumulative_trapezoid(ynp, initial=0.0),
        "`cumtrapz` with initial is broken",
    )
    check(
        nm.cumtrapz(y, x),
        sp_integrate.cumulative_trapezoid(ynp, xnp),
        "`cumtrapz` with x is broken",
    )


def test_ediff1d() raises:
    var np = Python.import_module("numpy")
    var a = nm.fromstring[nm.i64]("[1, 2, 4, 7, 0]")
    var anp = a.to_numpy()

    check_is_close(nm.ediff1d(a), np.ediff1d(anp), "`ediff1d` is broken")

    var to_begin = nm.fromstring[nm.i64]("[-99]")
    var to_end = nm.fromstring[nm.i64]("[88, 99]")
    check_is_close(
        nm.ediff1d(a, to_end=to_end.copy(), to_begin=to_begin.copy()),
        np.ediff1d(
            anp, to_end=to_end.to_numpy(), to_begin=to_begin.to_numpy()
        ),
        "`ediff1d` with to_begin/to_end is broken",
    )


def test_convolve() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(7)
    var v = nm.random.randn(4)
    var anp = a.to_numpy()
    var vnp = v.to_numpy()

    check_is_close(
        nm.convolve(a, v), np.convolve(anp, vnp), "`convolve` full is broken"
    )
    check_is_close(
        nm.convolve(a, v, mode="same"),
        np.convolve(anp, vnp, mode="same"),
        "`convolve` same is broken",
    )
    check_is_close(
        nm.convolve(a, v, mode="valid"),
        np.convolve(anp, vnp, mode="valid"),
        "`convolve` valid is broken",
    )

    with assert_raises():
        _ = nm.convolve(a, v, mode="bogus")


def test_interp() raises:
    var np = Python.import_module("numpy")
    var xp = nm.fromstring("[1.0, 2.0, 3.0, 4.0]")
    var fp = nm.fromstring("[3.0, 2.0, 0.0, 5.0]")
    var x = nm.fromstring("[0.0, 1.5, 2.5, 3.5, 5.0]")
    var xpnp = xp.to_numpy()
    var fpnp = fp.to_numpy()
    var xnp = x.to_numpy()

    check_is_close(
        nm.interp(x, xp, fp), np.interp(xnp, xpnp, fpnp), "`interp` is broken"
    )
    check_values_close(
        nm.interp(Float64(2.5), xp, fp),
        np.interp(2.5, xpnp, fpnp),
        "`interp` scalar is broken",
    )
    check_is_close(
        nm.interp(x, xp, fp, left=Float64(99.0), right=Float64(-1.0)),
        np.interp(xnp, xpnp, fpnp, left=99.0, right=-1.0),
        "`interp` left/right is broken",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
