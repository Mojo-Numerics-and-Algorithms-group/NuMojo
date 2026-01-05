import numojo as nm
from numojo.prelude import *
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true
from utils_for_test import (
    check,
    check_is_close,
    check_values_close,
    check_with_dtype,
)
from testing import TestSuite

# ===-----------------------------------------------------------------------===#
# Sums, products, differences
# ===-----------------------------------------------------------------------===#


def test_sum_prod():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2, 3, 4)
    var Anp = A.to_numpy()

    check_values_close(
        nm.sum(A),
        np.sum(Anp, axis=None),
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
        np.cumsum(Anp, axis=None),
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
        np.prod(Anp, axis=None),
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
        np.cumprod(Anp, axis=None),
        String("`cumprod` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.cumprod(A, axis=i),
            np.cumprod(Anp, axis=i),
            String("`cumprod` by axis {} fails.").format(i),
        )


def test_add_array():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check(nm.add[nm.f64](arr, 5.0), np.arange(0, 15) + 5, "Add array + scalar")
    check(
        nm.add[nm.f64](arr, arr),
        np.arange(0, 15) + np.arange(0, 15),
        "Add array + array",
    )


# def test_dunder_add_array():
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


# def test_dunder_sub_array():
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


def test_add_array_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 20)

    check(
        nm.add[nm.f64, backend = nm.routines.math._math_funcs.Vectorized](
            arr, 5.0
        ),
        np.arange(0, 20) + 5,
        "Add array + scalar",
    )
    check(
        nm.add[nm.f64, backend = nm.routines.math._math_funcs.Vectorized](
            arr, arr
        ),
        np.arange(0, 20) + np.arange(0, 20),
        "Add array + array",
    )


def test_sin():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[nm.f64](arr), np.sin(np.arange(0, 15)), "Add array + scalar"
    )


def test_sin_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[
            nm.f64,
            backend = nm.routines.math._math_funcs.Vectorized,
        ](arr),
        np.sin(np.arange(0, 15)),
        "Add array + scalar",
    )


fn test_extrema() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(10)
    var anp = a.to_numpy()
    var c = nm.random.randn(2, 3, 4)
    var cnp = c.to_numpy()
    var cf = c.reshape(c.shape, order="F")  # 3d array F order
    var cfnp = cf.to_numpy()

    # max
    check_values_close(nm.max(a), np.max(anp, axis=None), "`sort` 1d is broken")
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


fn test_misc() raises:
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


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
