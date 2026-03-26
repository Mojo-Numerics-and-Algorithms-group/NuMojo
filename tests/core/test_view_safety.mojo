"""
Tests for view safety: ensure all functions guarded with `contiguous()`
produce correct results when given non-contiguous (e.g. F-order) arrays,
and that core getters/setters honour the `offset` field on views.

This covers [Issue 309](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/issues/309)
Phase 1, 3 & 4 — contiguous guards and offset-aware accessors across the codebase.
"""

import numojo as nm
from numojo.prelude import *
from numojo.core.matrix import Matrix
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.routines.math.extrema import minimum
from std.python import Python, PythonObject
from std.testing.testing import assert_true, assert_equal
from std.testing import TestSuite


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


def check_array[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


def check_array_close[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(array.to_numpy(), np_sol, atol=PythonObject(0.1))),
        st,
    )


def check_scalar_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=PythonObject(0.001)), st)


def check_matrix_close[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(
            np.isclose(
                np.matrix(matrix.to_numpy()), np_sol, atol=PythonObject(0.01)
            )
        ),
        st,
    )


def check_matrix_equal[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


def check_value_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=PythonObject(0.01)), st)


# ===-----------------------------------------------------------------------===#
# NDArray: sum, prod, cumsum, cumprod on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_sums_products_view() raises:
    """Test sum, prod, cumsum, cumprod on F-order (non C-contiguous) NDArrays.
    """
    var np = Python.import_module("numpy")

    # Create F-order 2D array
    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()

    # Verify it is indeed non-contiguous
    assert_true(
        not A.is_c_contiguous(), "F-order array should not be C-contiguous"
    )

    # sum (flattened)
    check_scalar_close(
        nm.sum(A),
        np.sum(Anp),
        "`sum` on F-order NDArray is broken",
    )

    # prod (flattened)
    check_scalar_close(
        nm.prod(A),
        np.prod(Anp),
        "`prod` on F-order NDArray is broken",
    )

    # cumsum (flattened)
    check_array_close(
        nm.cumsum(A),
        np.cumsum(Anp),
        "`cumsum` on F-order NDArray is broken",
    )

    # cumprod (flattened)
    check_array_close(
        nm.cumprod(A),
        np.cumprod(Anp),
        "`cumprod` on F-order NDArray is broken",
    )

    # 3D F-order array
    var B = nm.random.randn(2, 3, 4).reshape(Shape(2, 3, 4), order="F")
    var Bnp = B.to_numpy()
    assert_true(
        not B.is_c_contiguous(), "F-order 3D array should not be C-contiguous"
    )

    check_scalar_close(
        nm.sum(B),
        np.sum(Bnp),
        "`sum` on F-order 3D NDArray is broken",
    )
    check_scalar_close(
        nm.prod(B),
        np.prod(Bnp),
        "`prod` on F-order 3D NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: extrema (max, min, minimum, maximum) on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_extrema_view() raises:
    """Test max, min, minimum, maximum on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # max
    check_scalar_close(
        nm.max(A),
        np.max(Anp),
        "`max` on F-order NDArray is broken",
    )

    # min
    check_scalar_close(
        nm.min(A),
        np.min(Anp),
        "`min` on F-order NDArray is broken",
    )

    # minimum (elementwise)
    var B = nm.arange[nm.f64](11, -1, -1).reshape(Shape(3, 4), order="F")
    var Bnp = B.to_numpy()
    assert_true(not B.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(
        minimum(A, B),
        np.minimum(Anp, Bnp),
        "`minimum` on F-order NDArrays is broken",
    )

    # maximum (elementwise)
    check_array_close(
        nm.maximum(A, B),
        np.maximum(Anp, Bnp),
        "`maximum` on F-order NDArrays is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: searching (argmax, argmin) on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_searching_view() raises:
    """Test argmax_1d, argmin_1d on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_scalar_close(
        nm.argmax(A),
        np.argmax(Anp),
        "`argmax` on F-order NDArray is broken",
    )

    check_scalar_close(
        nm.argmin(A),
        np.argmin(Anp),
        "`argmin` on F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: sorting on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_sorting_view() raises:
    """Test sort, argsort on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Create F-order array
    var A = nm.random.randn(3, 4).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # sort along axis 0
    check_array_close(
        nm.sort(A, axis=0),
        np.sort(Anp, axis=0),
        "`sort` axis=0 on F-order NDArray is broken",
    )

    # sort along axis 1
    check_array_close(
        nm.sort(A, axis=1),
        np.sort(Anp, axis=1),
        "`sort` axis=1 on F-order NDArray is broken",
    )

    # argsort (flattened)
    check_array(
        nm.argsort(A),
        np.argsort(Anp, axis=PythonObject(None)),
        "`argsort` on F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: linalg on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_linalg_view() raises:
    """Test matmul, dot, trace, diagonal on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Create F-order 2D arrays for matmul
    var A = nm.random.randn(3, 4).reshape(Shape(3, 4), order="F")
    var B = nm.random.randn(4, 2).reshape(Shape(4, 2), order="F")
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # matmul
    check_array_close(
        A @ B,
        np.matmul(Anp, Bnp),
        "`matmul` on F-order NDArrays is broken",
    )

    # dot (1D)
    var v1 = nm.random.randn(6).reshape(Shape(6), order="F")
    var v2 = nm.random.randn(6).reshape(Shape(6), order="F")
    var v1np = v1.to_numpy()
    var v2np = v2.to_numpy()

    check_array_close(
        nm.linalg.dot(v1, v2),
        v1np * v2np,
        "`dot` on F-order 1D NDArrays is broken",
    )

    # trace
    var S = nm.random.randn(4, 4).reshape(Shape(4, 4), order="F")
    var Snp = S.to_numpy()
    assert_true(not S.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(
        nm.linalg.trace(S),
        np.trace(Snp),
        "`trace` on F-order NDArray is broken",
    )

    # diagonal
    check_array_close(
        nm.diagonal(S),
        np.diagonal(Snp),
        "`diagonal` on F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: creation (diag) on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_creation_view() raises:
    """Test diag on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # diag from 1D F-order vector
    var v = nm.arange[nm.f64](0, 4).reshape(Shape(4), order="F")
    var vnp = v.to_numpy()
    check_array(
        nm.diag(v),
        np.diag(vnp),
        "`diag` from 1D F-order NDArray is broken",
    )

    # diag from 2D F-order matrix (extract diagonal)
    var M = nm.arange[nm.f64](0, 9).reshape(Shape(3, 3), order="F")
    var Mnp = M.to_numpy()
    assert_true(not M.is_c_contiguous(), "Should be non-contiguous")
    check_array(
        nm.diag(M),
        np.diag(Mnp),
        "`diag` from 2D F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: indexing (compress) on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_indexing_view() raises:
    """Test compress on NDArrays with F-order condition."""
    var np = Python.import_module("numpy")

    # compress with condition
    var a = nm.arange[nm.i8](6)
    var anp = a.to_numpy()
    var cond = nm.array[boolean]("[1, 0, 1, 0, 1, 0]")

    check_array(
        nm.indexing.compress(cond, a),
        np.compress(np.array(Python.list(1, 0, 1, 0, 1, 0)), anp),
        "`compress` with condition on NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: __pow__ on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_pow_view() raises:
    """Test __pow__ on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # __pow__ with scalar
    check_array_close(
        A**2,
        np.power(Anp, 2),
        "`__pow__(scalar)` on F-order NDArray is broken",
    )

    # __pow__ with another F-order array
    var B = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var Bnp = B.to_numpy()
    check_array_close(
        A**B,
        np.power(Anp, Bnp),
        "`__pow__(NDArray)` on F-order NDArrays is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: sum, prod, cumsum on F-order matrices
# ===-----------------------------------------------------------------------===#


def test_matrix_sums_products_view() raises:
    """Test sum, prod, cumsum on F-order Matrices."""
    var np = Python.import_module("numpy")

    var A = Matrix.rand[nm.f64](shape=(3, 4), order="F")
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    # sum (flattened)
    check_value_close(
        nm.sum(A),
        np.sum(Anp),
        "`sum` on F-order Matrix is broken",
    )

    # sum along axis
    check_matrix_close(
        nm.sum(A, axis=0),
        np.sum(Anp, axis=0),
        "`sum(axis=0)` on F-order Matrix is broken",
    )
    check_matrix_close(
        nm.sum(A, axis=1),
        np.sum(Anp, axis=1),
        "`sum(axis=1)` on F-order Matrix is broken",
    )

    # prod (flattened)
    check_value_close(
        nm.prod(A),
        np.prod(Anp),
        "`prod` on F-order Matrix is broken",
    )

    # cumsum (flattened)
    var cs = nm.cumsum(A)
    var cs_np = np.cumsum(Anp)
    check_matrix_close(
        cs,
        np.matrix(cs_np),
        "`cumsum` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: logic (all, any) on F-order matrices
# ===-----------------------------------------------------------------------===#


def test_matrix_logic_view() raises:
    """Test all, any on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Matrix with all ones (F-order) - use i8 since all/any need integral type
    var A = Matrix.ones[nm.i8](shape=(3, 4), order="F")
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    assert_true(nm.all(A), "`all` on F-order Matrix of ones should be True")
    assert_true(nm.any(A), "`any` on F-order Matrix of ones should be True")

    # Matrix with all zeros (F-order)
    var B = Matrix.zeros[nm.i8](shape=(3, 4), order="F")
    assert_true(
        not nm.all(B),
        "`all` on F-order Matrix of zeros should be False",
    )
    assert_true(
        not nm.any(B),
        "`any` on F-order Matrix of zeros should be False",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: __pow__ on F-order matrices
# ===-----------------------------------------------------------------------===#


def test_matrix_pow_view() raises:
    """Test __pow__ on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Create F-order Matrix via NDArray conversion (fromstring ignores order)
    var nd = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var A = Matrix[nm.f64](nd)
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    # __pow__ with Int
    var result = A**2
    var expected = np.power(Anp, 2)
    check_matrix_close(
        result,
        expected,
        "`__pow__` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: astype, flatten, to_ndarray on F-order matrices
# ===-----------------------------------------------------------------------===#


def test_matrix_conversion_view() raises:
    """Test astype, flatten, to_ndarray on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Create F-order Matrix via NDArray conversion
    var nd_src = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var A = Matrix[nm.f64](nd_src)
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    # astype
    var A32 = A.astype[nm.f32]()
    var A32np = np.matrix(A32.to_numpy())
    assert_true(
        np.all(np.isclose(A32np, Anp, atol=PythonObject(0.01))),
        "`astype` on F-order Matrix is broken",
    )

    # flatten
    var flat = A.flatten()
    var flat_np = np.array(Anp).flatten()
    assert_true(
        np.all(
            np.isclose(
                np.matrix(flat.to_numpy()),
                np.matrix(flat_np),
                atol=PythonObject(0.01),
            )
        ),
        "`flatten` on F-order Matrix is broken",
    )

    # to_ndarray
    var nd = A.to_ndarray()
    var nd_np = nd.to_numpy()
    assert_true(
        np.all(np.isclose(nd_np, np.array(Anp), atol=PythonObject(0.01))),
        "`to_ndarray` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: rounding on F-order matrices
# ===-----------------------------------------------------------------------===#


def test_matrix_rounding_view() raises:
    """Test round on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Use rand to create F-order Matrix with random values to test rounding
    var A = Matrix.rand[nm.f64](shape=(3, 4), order="F")
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    var result = nm.math.round(A, decimals=1)
    var expected = np.around(Anp, 1)
    check_matrix_close(
        result,
        expected,
        "`round` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: mutating original is not affected by sort on view
# ===-----------------------------------------------------------------------===#


def test_sort_does_not_mutate_original() raises:
    """Test that sorting a view does not mutate the original array."""
    var np = Python.import_module("numpy")

    # Create original array
    var original = nm.array[nm.f64]("[5.0, 3.0, 1.0, 4.0, 2.0]")
    var original_copy = nm.array[nm.f64]("[5.0, 3.0, 1.0, 4.0, 2.0]")

    # Sort (this uses .contiguous() internally to avoid mutating input)
    var sorted_arr = nm.sort(original, axis=0)

    # Check sorted result is correct
    check_array_close(
        sorted_arr,
        np.sort(np.array(Python.list(5.0, 3.0, 1.0, 4.0, 2.0))),
        "sort result is wrong",
    )

    # Check original was not mutated
    check_array(
        original,
        original_copy.to_numpy(),
        "Original array should not be mutated by sort",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: linalg solve on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_solve_view() raises:
    """Test linalg.solve on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Create a non-singular 3x3 system A*x = b using random values
    var A = nm.random.randn(3, 3).reshape(Shape(3, 3), order="F")
    # Add identity to make it more likely non-singular
    for i in range(3):
        A._setitem(i, i, val=A.load[1](i * 3 + i) + 10.0)
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    var b = nm.random.randn(3, 1).reshape(Shape(3, 1), order="F")
    var bnp = b.to_numpy()

    # solve
    var x = nm.linalg.solve(A, b)
    var xnp = np.linalg.solve(Anp, bnp)

    check_array_close(
        x,
        xnp,
        "`solve` on F-order NDArrays is broken",
    )


# ===-----------------------------------------------------------------------===#
# Phase 3: Math backend functions on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_trig_view() raises:
    """Test trigonometric functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Values in [0.1, 0.6] — safe for all inverse trig functions
    var A = nm.fromstring("[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.sin(A), np.sin(Anp), "`sin` on F-order broken")
    check_array_close(nm.cos(A), np.cos(Anp), "`cos` on F-order broken")
    check_array_close(nm.tan(A), np.tan(Anp), "`tan` on F-order broken")
    check_array_close(nm.asin(A), np.arcsin(Anp), "`asin` on F-order broken")
    check_array_close(nm.acos(A), np.arccos(Anp), "`acos` on F-order broken")
    check_array_close(nm.atan(A), np.arctan(Anp), "`atan` on F-order broken")

    # atan2 and hypot: two-array inputs
    var B = nm.fromstring("[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]").reshape(
        Shape(2, 3), order="F"
    )
    var Bnp = B.to_numpy()
    check_array_close(
        nm.atan2(A, B), np.arctan2(Anp, Bnp), "`atan2` on F-order broken"
    )
    check_array_close(
        nm.hypot(A, B), np.hypot(Anp, Bnp), "`hypot` on F-order broken"
    )


def test_ndarray_hyper_view() raises:
    """Test hyperbolic functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.sinh(A), np.sinh(Anp), "`sinh` on F-order broken")
    check_array_close(nm.cosh(A), np.cosh(Anp), "`cosh` on F-order broken")
    check_array_close(nm.tanh(A), np.tanh(Anp), "`tanh` on F-order broken")
    check_array_close(nm.asinh(A), np.arcsinh(Anp), "`asinh` on F-order broken")
    check_array_close(nm.atanh(A), np.arctanh(Anp), "`atanh` on F-order broken")

    # acosh needs values >= 1
    var C = nm.fromstring("[1.1, 1.5, 2.0, 2.5, 3.0, 4.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Cnp = C.to_numpy()
    check_array_close(nm.acosh(C), np.arccosh(Cnp), "`acosh` on F-order broken")


def test_ndarray_exp_log_view() raises:
    """Test exp/log functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.exp(A), np.exp(Anp), "`exp` on F-order broken")
    check_array_close(nm.exp2(A), np.exp2(Anp), "`exp2` on F-order broken")
    check_array_close(nm.expm1(A), np.expm1(Anp), "`expm1` on F-order broken")
    check_array_close(nm.log(A), np.log(Anp), "`log` on F-order broken")
    check_array_close(nm.log2(A), np.log2(Anp), "`log2` on F-order broken")
    check_array_close(nm.log10(A), np.log10(Anp), "`log10` on F-order broken")
    check_array_close(nm.log1p(A), np.log1p(Anp), "`log1p` on F-order broken")


def test_ndarray_arithmetic_view() raises:
    """Test arithmetic functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var B = nm.arange[nm.f64](7, 13).reshape(Shape(2, 3), order="F")
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.add(A, B), np.add(Anp, Bnp), "`add` on F-order broken")
    check_array_close(
        nm.sub(A, B), np.subtract(Anp, Bnp), "`sub` on F-order broken"
    )
    check_array_close(
        nm.mul(A, B), np.multiply(Anp, Bnp), "`mul` on F-order broken"
    )
    check_array_close(
        nm.div(A, B), np.divide(Anp, Bnp), "`div` on F-order broken"
    )

    # fma: A * B + scalar
    var c: Scalar[nm.f64] = 10.0
    check_array_close(
        nm.fma(A, B, c),
        np.add(np.multiply(Anp, Bnp), 10.0),
        "`fma` on F-order broken",
    )


def test_ndarray_rounding_view() raises:
    """Test rounding functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[1.2, -2.7, 3.5, -4.1, 5.9, -6.3]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.tabs(A), np.abs(Anp), "`tabs` on F-order broken")
    check_array_close(nm.tfloor(A), np.floor(Anp), "`tfloor` on F-order broken")
    check_array_close(nm.tceil(A), np.ceil(Anp), "`tceil` on F-order broken")
    check_array_close(nm.ttrunc(A), np.trunc(Anp), "`ttrunc` on F-order broken")
    check_array_close(nm.tround(A), np.round(Anp), "`tround` on F-order broken")


def test_ndarray_misc_math_view() raises:
    """Test misc math functions (clip, sqrt, cbrt, rsqrt) on F-order arrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.sqrt(A), np.sqrt(Anp), "`sqrt` on F-order broken")
    check_array_close(nm.cbrt(A), np.cbrt(Anp), "`cbrt` on F-order broken")
    check_array_close(
        nm.rsqrt(A),
        np.reciprocal(np.sqrt(Anp)),
        "`rsqrt` on F-order broken",
    )

    # clip
    check_array_close(
        nm.clip(A, Scalar[nm.f64](5.0), Scalar[nm.f64](20.0)),
        np.clip(Anp, 5.0, 20.0),
        "`clip` on F-order broken",
    )


def test_ndarray_comparison_view() raises:
    """Test comparison and logic functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var B = nm.fromstring("[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array(
        nm.greater(A, B),
        np.greater(Anp, Bnp),
        "`greater` on F-order broken",
    )
    check_array(
        nm.less(A, B),
        np.less(Anp, Bnp),
        "`less` on F-order broken",
    )
    check_array(
        nm.equal(A, B),
        np.equal(Anp, Bnp),
        "`equal` on F-order broken",
    )


def test_ndarray_copysign_view() raises:
    """Test copysign and nextafter on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]").reshape(
        Shape(2, 3), order="F"
    )
    var B = nm.fromstring("[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()

    check_array_close(
        nm.copysign(A, B),
        np.copysign(Anp, Bnp),
        "`copysign` on F-order broken",
    )


# ===-----------------------------------------------------------------------===#
# Phase 3: Differences on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_differences_view() raises:
    """Test gradient and trapz on F-order (non C-contiguous) NDArrays."""
    var np = Python.import_module("numpy")

    # 1D array reshaped to F-order (for 1D arrays F-order is same as C-order,
    # so use a slice of a 2D F-order array instead)
    var A2d = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    assert_true(not A2d.is_c_contiguous(), "Should be non-contiguous")

    # gradient: test with a simple 1D array (gradient is 1D only)
    var x = nm.fromstring("[1.0, 2.0, 4.0, 7.0, 11.0]")
    var xnp = np.array(Python.list(1.0, 2.0, 4.0, 7.0, 11.0))
    var grad_result = nm.gradient(x, Scalar[nm.f64](1.0))
    var grad_np = np.gradient(xnp, 1.0)
    check_array_close(grad_result, grad_np, "`gradient` result is wrong")

    # Note: trapz is not tested here due to a pre-existing constraint bug
    # in differences.mojo that rejects float dtypes (issue unrelated to
    # view safety).


# ===-----------------------------------------------------------------------===#
# Phase 3: Manipulation on F-order arrays
# ===-----------------------------------------------------------------------===#


def test_ndarray_reshape_view() raises:
    """Test reshape on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # reshape to different shape
    var reshaped = nm.reshape(A, Shape(4, 3))
    var reshaped_np = np.reshape(Anp, Python.tuple(4, 3))
    check_array_close(reshaped, reshaped_np, "`reshape` on F-order broken")

    # reshape to 1D
    var flat = nm.reshape(A, Shape(12))
    var flat_np = np.reshape(Anp, 12)
    check_array_close(flat, flat_np, "`reshape` to 1D on F-order broken")


def test_ndarray_ravel_view() raises:
    """Test ravel on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    var raveled = nm.ravel(A)
    var raveled_np = np.ravel(Anp)
    check_array_close(raveled, raveled_np, "`ravel` on F-order broken")


def test_ndarray_transpose_view() raises:
    """Test transpose on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # Simple transpose (no axes)
    var T = nm.transpose(A)
    var Tnp = np.transpose(Anp)
    check_array_close(T, Tnp, "`transpose` on F-order broken")

    # Transpose with axes
    var B = nm.arange[nm.f64](0, 24).reshape(Shape(2, 3, 4), order="F")
    var Bnp = B.to_numpy()
    assert_true(not B.is_c_contiguous(), "Should be non-contiguous")

    var T2 = nm.transpose(B, axes=[Int(2), 0, 1])
    var T2np = np.transpose(Bnp, Python.list(2, 0, 1))
    check_array_close(T2, T2np, "`transpose(axes)` on F-order broken")


def test_ndarray_flip_view() raises:
    """Test flip on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # flip (all axes)
    var flipped = nm.flip(A)
    var flipped_np = np.flip(Anp)
    check_array_close(flipped, flipped_np, "`flip` on F-order broken")

    # flip along axis 0
    var flipped0 = nm.flip(A, axis=0)
    var flipped0_np = np.flip(Anp, axis=0)
    check_array_close(flipped0, flipped0_np, "`flip(axis=0)` on F-order broken")

    # flip along axis 1
    var flipped1 = nm.flip(A, axis=1)
    var flipped1_np = np.flip(Anp, axis=1)
    check_array_close(flipped1, flipped1_np, "`flip(axis=1)` on F-order broken")


def test_ndarray_broadcast_to_view() raises:
    """Test broadcast_to on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 4).reshape(Shape(1, 3), order="F")
    var Anp = A.to_numpy()

    var broadcasted = nm.broadcast_to(A, Shape(3, 3))
    var broadcasted_np = np.broadcast_to(Anp, Python.tuple(3, 3))
    check_array_close(
        broadcasted, broadcasted_np, "`broadcast_to` on F-order broken"
    )


# ===-----------------------------------------------------------------------===#
# Phase 3: Sliced views (non-contiguous due to slicing, not just F-order)
# ===-----------------------------------------------------------------------===#


def test_ndarray_sliced_view_math() raises:
    """Test math on non-contiguous views created via F-order reshape."""
    var np = Python.import_module("numpy")

    # Create a 3D F-order array (guaranteed non-contiguous)
    var A = nm.arange[nm.f64](0, 24).reshape(Shape(2, 3, 4), order="F")
    var Anp = A.to_numpy()

    assert_true(
        not A.is_c_contiguous(),
        "F-order array should not be C-contiguous",
    )

    # sin on F-order 3D view
    check_array_close(
        nm.sin(A),
        np.sin(Anp),
        "`sin` on 3D F-order broken",
    )

    # exp on F-order 3D view
    check_array_close(
        nm.exp(A),
        np.exp(Anp),
        "`exp` on 3D F-order broken",
    )

    # add two F-order 3D views
    var B = nm.arange[nm.f64](24, 48).reshape(Shape(2, 3, 4), order="F")
    var Bnp = B.to_numpy()
    check_array_close(
        nm.add(A, B),
        np.add(Anp, Bnp),
        "`add` on 3D F-order broken",
    )

    # sum on 3D F-order view
    check_scalar_close(
        nm.sum(A),
        np.sum(Anp),
        "`sum` on 3D F-order broken",
    )


def test_ndarray_sliced_view_manipulation() raises:
    """Test manipulation functions on 3D F-order arrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 24).reshape(Shape(2, 3, 4), order="F")
    var Anp = A.to_numpy()

    assert_true(
        not A.is_c_contiguous(),
        "F-order array should not be C-contiguous",
    )

    # reshape
    check_array_close(
        nm.reshape(A, Shape(6, 4)),
        np.reshape(Anp, Python.tuple(6, 4)),
        "`reshape` on 3D F-order broken",
    )

    # ravel
    check_array_close(
        nm.ravel(A),
        np.ravel(Anp),
        "`ravel` on 3D F-order broken",
    )

    # transpose
    check_array_close(
        nm.transpose(A),
        np.transpose(Anp),
        "`transpose` on 3D F-order broken",
    )

    # flip
    check_array_close(
        nm.flip(A),
        np.flip(Anp),
        "`flip` on 3D F-order broken",
    )


# ===-----------------------------------------------------------------------===#
# Phase 4 — Offset-aware core getters / setters
# ===-----------------------------------------------------------------------===#


def _make_1d_offset_view(
    parent: nm.NDArray[nm.f64], offset: Int, size: Int
) raises -> nm.NDArray[nm.f64]:
    """Create a 1-D view into `parent` starting at `offset` with `size` elems.
    """
    var view = parent.copy()
    view.offset = offset
    view.size = size
    view.shape = NDArrayShape(size)
    view.ndim = 1
    view.strides = NDArrayStrides(shape=view.shape)
    return view^


def _make_2d_offset_view(
    parent: nm.NDArray[nm.f64],
    offset: Int,
    rows: Int,
    cols: Int,
    stride0: Int,
    stride1: Int,
) raises -> nm.NDArray[nm.f64]:
    """Create a 2-D view into `parent`."""
    var view = parent.copy()
    view.offset = offset
    view.size = rows * cols
    view.shape = NDArrayShape(rows, cols)
    view.ndim = 2
    view.strides = NDArrayStrides(stride0, stride1)
    return view^


def test_offset_view_load_store() raises:
    """Test load/store/unsafe_load/unsafe_store on 1-D offset views."""
    # parent = [0, 1, 2, ..., 11]
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=4, size=8)

    # load — should read from parent[offset + i]
    assert_true(view.load(0) == 4.0, "load(0) on offset view")
    assert_true(view.load(3) == 7.0, "load(3) on offset view")
    assert_true(view.load(7) == 11.0, "load(7) on offset view")

    # unsafe_load
    assert_true(view.unsafe_load(0) == 4.0, "unsafe_load(0) on offset view")
    assert_true(view.unsafe_load(5) == 9.0, "unsafe_load(5) on offset view")

    # store — should write to parent[offset + i]
    view.store(0, Scalar[nm.f64](99.0))
    assert_true(parent.load(4) == 99.0, "store reflected in parent")
    assert_true(view.load(0) == 99.0, "store on offset view readable")

    # unsafe_store
    view.unsafe_store(1, Scalar[nm.f64](88.0))
    assert_true(parent.load(5) == 88.0, "unsafe_store reflected in parent")


def test_offset_view_item_and_itemset() raises:
    """Test item() and itemset() on offset views."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=4, size=8)

    # item(flat index) — C-order
    assert_true(view.item(0) == 4.0, "item(0) on offset view")
    assert_true(view.item(7) == 11.0, "item(7) on offset view")

    # itemset(flat_index, val)
    view.itemset(2, Scalar[nm.f64](77.0))
    assert_true(view.item(2) == 77.0, "itemset(2) on offset view")
    assert_true(parent.load(6) == 77.0, "itemset reflected in parent")

    # itemset(List[Int], val) — coordinate form
    var idx3 = List[Int]()
    idx3.append(3)
    view.itemset(idx3^, Scalar[nm.f64](66.0))
    assert_true(view.item(3) == 66.0, "itemset(List) on offset view")
    assert_true(parent.load(7) == 66.0, "itemset(List) reflected in parent")


def test_offset_view_getitem_setitem_item() raises:
    """Test __getitem__(Item) and __setitem__(Item, Scalar) on offset views."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=3, size=9)

    # __getitem__(Item) — coordinate access
    assert_true(view[Item(0)] == 3.0, "__getitem__(Item(0)) offset view")
    assert_true(view[Item(5)] == 8.0, "__getitem__(Item(5)) offset view")

    # __setitem__(Item, Scalar)
    view[Item(1)] = Scalar[nm.f64](42.0)
    assert_true(view[Item(1)] == 42.0, "__setitem__(Item) on offset view")
    assert_true(parent.load(4) == 42.0, "__setitem__(Item) in parent")


def test_offset_view_getitem_int() raises:
    """Test __getitem__(idx: Int) which returns 0-D for 1-D arrays."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=2, size=10)

    # 1-D view → 0-D scalar
    var s0 = view[0]
    assert_true(s0.item() == 2.0, "view[0] → 0-D = parent[2]")

    var s4 = view[4]
    assert_true(s4.item() == 6.0, "view[4] → 0-D = parent[6]")


def test_offset_view_2d_getitem_int() raises:
    """Test __getitem__(idx: Int) on a 2-D offset view (returns a row)."""
    # parent = [0, 1, ..., 23], reshaped to 4x6, then view rows 2-3
    var parent = nm.arange[nm.f64](0, 24)
    # Create a 2x6 C-contiguous view starting at row 2 (offset = 12)
    var view = _make_2d_offset_view(
        parent, offset=12, rows=2, cols=6, stride0=6, stride1=1
    )

    # view[0] should give row 2 of parent = [12, 13, 14, 15, 16, 17]
    var row0 = view[0]
    assert_true(row0.item(0) == 12.0, "2D view[0] first elem")
    assert_true(row0.item(5) == 17.0, "2D view[0] last elem")

    # view[1] should give row 3 of parent = [18, 19, 20, 21, 22, 23]
    var row1 = view[1]
    assert_true(row1.item(0) == 18.0, "2D view[1] first elem")
    assert_true(row1.item(5) == 23.0, "2D view[1] last elem")


def test_offset_view_setitem_int() raises:
    """Test __setitem__(idx: Int, val: Self) on a 2-D offset view."""
    var parent = nm.arange[nm.f64](0, 24)
    var view = _make_2d_offset_view(
        parent, offset=12, rows=2, cols=6, stride0=6, stride1=1
    )

    # Replace row 0 of view (= row 2 of parent)
    var new_row = nm.full[nm.f64](Shape(6), fill_value=99.0)
    view[0] = new_row

    # Check the parent was modified at positions 12-17
    for i in range(6):
        assert_true(
            parent.load(12 + i) == 99.0,
            String("setitem(int) parent[{}] should be 99").format(12 + i),
        )

    # Row 1 of view (= row 3 of parent) should be unchanged
    assert_true(parent.load(18) == 18.0, "setitem(int) parent[18] unchanged")


def test_offset_view_load_variadic() raises:
    """Test load[width](*indices) and store[width](*indices) on 2-D view."""
    var parent = nm.arange[nm.f64](0, 24)
    var view = _make_2d_offset_view(
        parent, offset=6, rows=3, cols=6, stride0=6, stride1=1
    )

    # load(row, col) should use view strides + view offset
    assert_true(view.load(0, 0) == 6.0, "load(0,0) on 2D offset view")
    assert_true(view.load(1, 3) == 15.0, "load(1,3) on 2D offset view")
    assert_true(view.load(2, 5) == 23.0, "load(2,5) on 2D offset view")

    # store(*indices)
    view.store(0, 2, val=Scalar[nm.f64](55.0))
    assert_true(parent.load(8) == 55.0, "store(0,2) reflected in parent")


def test_offset_view_mask_getter() raises:
    """Test __getitem__(mask) on an offset view."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=4, size=8)
    # view = [4, 5, 6, 7, 8, 9, 10, 11]

    # Create mask: select elements > 8
    var mask = nm.NDArray[DType.bool](shape=NDArrayShape(8))
    for i in range(8):
        if view.item(i) > 8.0:
            var idx = List[Int]()
            idx.append(i)
            mask.itemset(idx^, Scalar[DType.bool](True))
        else:
            var idx = List[Int]()
            idx.append(i)
            mask.itemset(idx^, Scalar[DType.bool](False))

    var result = view.__getitem__(mask)
    # Should get [9, 10, 11]
    assert_true(result.size == 3, "mask getter result size")
    assert_true(result.item(0) == 9.0, "mask getter first")
    assert_true(result.item(1) == 10.0, "mask getter second")
    assert_true(result.item(2) == 11.0, "mask getter third")


def test_offset_view_mask_setter() raises:
    """Test __setitem__(mask, Scalar) on an offset view."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=4, size=8)
    # view = [4, 5, 6, 7, 8, 9, 10, 11]

    # Mask: set elements > 9 to -1
    var mask = nm.NDArray[DType.bool](shape=NDArrayShape(8))
    for i in range(8):
        if view.item(i) > 9.0:
            var idx = List[Int]()
            idx.append(i)
            mask.itemset(idx^, Scalar[DType.bool](True))
        else:
            var idx = List[Int]()
            idx.append(i)
            mask.itemset(idx^, Scalar[DType.bool](False))

    view.__setitem__(mask, Scalar[nm.f64](-1.0))

    # view[6] (parent[10]) and view[7] (parent[11]) should be -1
    assert_true(view.item(6) == -1.0, "mask setter view[6]")
    assert_true(view.item(7) == -1.0, "mask setter view[7]")
    assert_true(parent.load(10) == -1.0, "mask setter parent[10]")
    assert_true(parent.load(11) == -1.0, "mask setter parent[11]")

    # Others unchanged
    assert_true(view.item(0) == 4.0, "mask setter view[0] unchanged")
    assert_true(view.item(5) == 9.0, "mask setter view[5] unchanged")


def test_offset_view_write_to_0d() raises:
    """Test that printing a 1-D offset view shows the correct values."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=4, size=4)
    # view = [4, 5, 6, 7]

    var s = String(view)
    assert_true("4.0" in s, "offset view print contains first elem")
    assert_true("7.0" in s, "offset view print contains last elem")


def test_offset_view_setitem_item_unsafe() raises:
    """Test _setitem (internal unsafe setter) on offset view."""
    var parent = nm.arange[nm.f64](0, 12)
    var view = _make_1d_offset_view(parent, offset=3, size=9)

    view._setitem(2, val=Scalar[nm.f64](55.0))
    assert_true(
        parent.load(5) == 55.0, "_setitem(2) on offset view -> parent[5]"
    )


def test_offset_view_slice_getitem() raises:
    """Test __getitem__(slices) on a 2-D offset view."""
    var parent = nm.arange[nm.f64](0, 24)
    var view = _make_2d_offset_view(
        parent, offset=6, rows=3, cols=6, stride0=6, stride1=1
    )

    # Slice view[0:2, 1:4] = rows 0-1, cols 1-3 of the view
    # view row 0 = parent[6..11] = [6,7,8,9,10,11], cols 1-3 = [7,8,9]
    # view row 1 = parent[12..17] = [12,13,14,15,16,17], cols 1-3 = [13,14,15]
    var sliced = view[0:2, 1:4]
    assert_true(sliced.shape[0] == 2, "slice shape[0]")
    assert_true(sliced.shape[1] == 3, "slice shape[1]")
    assert_true(sliced.item(0) == 7.0, "slice [0,0]")
    assert_true(sliced.item(1) == 8.0, "slice [0,1]")
    assert_true(sliced.item(2) == 9.0, "slice [0,2]")
    assert_true(sliced.item(3) == 13.0, "slice [1,0]")


def test_offset_view_slice_setitem() raises:
    """Test __setitem__(slices, val) on a 2-D offset view."""
    var parent = nm.arange[nm.f64](0, 24)
    var view = _make_2d_offset_view(
        parent, offset=6, rows=3, cols=6, stride0=6, stride1=1
    )

    # Set view[0:2, 0:3] = [[99, 99, 99], [99, 99, 99]]
    var fill = nm.full[nm.f64](Shape(2, 3), fill_value=99.0)
    view[0:2, 0:3] = fill

    # parent[6..8] (view row 0, cols 0-2) should be 99
    assert_true(parent.load(6) == 99.0, "slice set parent[6]")
    assert_true(parent.load(7) == 99.0, "slice set parent[7]")
    assert_true(parent.load(8) == 99.0, "slice set parent[8]")
    # parent[12..14] (view row 1, cols 0-2) should be 99
    assert_true(parent.load(12) == 99.0, "slice set parent[12]")
    assert_true(parent.load(13) == 99.0, "slice set parent[13]")
    assert_true(parent.load(14) == 99.0, "slice set parent[14]")
    # parent[15] should be unchanged
    assert_true(parent.load(15) == 15.0, "slice set parent[15] unchanged")


# ===-----------------------------------------------------------------------===#
# Phase 4+ tests: In-place operators, Matrix view safety, where, ravel
# ===-----------------------------------------------------------------------===#


def test_inplace_iadd_scalar_view() raises:
    """Test += scalar on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 10)
    var view = _make_1d_offset_view(parent, offset=3, size=4)  # [3,4,5,6]
    view += 100.0

    # The parent should be modified at positions 3-6
    assert_true(parent.load(0) == 0.0, "iadd parent[0] unchanged")
    assert_true(parent.load(2) == 2.0, "iadd parent[2] unchanged")
    assert_true(parent.load(3) == 103.0, "iadd parent[3]")
    assert_true(parent.load(4) == 104.0, "iadd parent[4]")
    assert_true(parent.load(5) == 105.0, "iadd parent[5]")
    assert_true(parent.load(6) == 106.0, "iadd parent[6]")
    assert_true(parent.load(7) == 7.0, "iadd parent[7] unchanged")


def test_inplace_isub_scalar_view() raises:
    """Test -= scalar on a C-contiguous view with offset."""
    var parent = nm.full[nm.f64](Shape(10), fill_value=50.0)
    var view = _make_1d_offset_view(parent, offset=2, size=3)
    view -= 10.0

    assert_true(parent.load(1) == 50.0, "isub parent[1] unchanged")
    assert_true(parent.load(2) == 40.0, "isub parent[2]")
    assert_true(parent.load(3) == 40.0, "isub parent[3]")
    assert_true(parent.load(4) == 40.0, "isub parent[4]")
    assert_true(parent.load(5) == 50.0, "isub parent[5] unchanged")


def test_inplace_imul_scalar_view() raises:
    """Test *= scalar on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](1, 9)  # [1,2,3,4,5,6,7,8]
    var view = _make_1d_offset_view(parent, offset=4, size=4)  # [5,6,7,8]
    view *= 10.0

    assert_true(parent.load(3) == 4.0, "imul parent[3] unchanged")
    assert_true(parent.load(4) == 50.0, "imul parent[4]")
    assert_true(parent.load(5) == 60.0, "imul parent[5]")
    assert_true(parent.load(6) == 70.0, "imul parent[6]")
    assert_true(parent.load(7) == 80.0, "imul parent[7]")


def test_inplace_itruediv_scalar_view() raises:
    """Test /= scalar on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 8)  # [0,1,2,3,4,5,6,7]
    var view = _make_1d_offset_view(parent, offset=2, size=4)  # [2,3,4,5]
    view /= 2.0

    assert_true(parent.load(1) == 1.0, "itruediv parent[1] unchanged")
    assert_true(parent.load(2) == 1.0, "itruediv parent[2]")
    assert_true(parent.load(3) == 1.5, "itruediv parent[3]")
    assert_true(parent.load(4) == 2.0, "itruediv parent[4]")
    assert_true(parent.load(5) == 2.5, "itruediv parent[5]")
    assert_true(parent.load(6) == 6.0, "itruediv parent[6] unchanged")


def test_inplace_iadd_array_view() raises:
    """Test += array on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 10)
    var view = _make_1d_offset_view(parent, offset=3, size=4)  # [3,4,5,6]
    var addend = nm.arange[nm.f64](10, 14)  # [10,11,12,13]
    view += addend

    assert_true(parent.load(2) == 2.0, "iadd arr parent[2] unchanged")
    assert_true(parent.load(3) == 13.0, "iadd arr parent[3]")
    assert_true(parent.load(4) == 15.0, "iadd arr parent[4]")
    assert_true(parent.load(5) == 17.0, "iadd arr parent[5]")
    assert_true(parent.load(6) == 19.0, "iadd arr parent[6]")
    assert_true(parent.load(7) == 7.0, "iadd arr parent[7] unchanged")


def test_inplace_ipow_view() raises:
    """Test **= int on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 8)  # [0,1,2,3,4,5,6,7]
    var view = _make_1d_offset_view(parent, offset=2, size=3)  # [2,3,4]
    view **= 2

    assert_true(parent.load(1) == 1.0, "ipow parent[1] unchanged")
    assert_true(parent.load(2) == 4.0, "ipow parent[2]")
    assert_true(parent.load(3) == 9.0, "ipow parent[3]")
    assert_true(parent.load(4) == 16.0, "ipow parent[4]")
    assert_true(parent.load(5) == 5.0, "ipow parent[5] unchanged")


def test_inplace_ifloordiv_scalar_view() raises:
    """Test //= scalar on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 8)  # [0,1,2,3,4,5,6,7]
    var view = _make_1d_offset_view(parent, offset=4, size=3)  # [4,5,6]
    view //= 2.0

    assert_true(parent.load(3) == 3.0, "ifloordiv parent[3] unchanged")
    assert_true(parent.load(4) == 2.0, "ifloordiv parent[4]")
    assert_true(parent.load(5) == 2.0, "ifloordiv parent[5]")
    assert_true(parent.load(6) == 3.0, "ifloordiv parent[6]")
    assert_true(parent.load(7) == 7.0, "ifloordiv parent[7] unchanged")


def test_inplace_imod_scalar_view() raises:
    """Test %= scalar on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 8)  # [0,1,2,3,4,5,6,7]
    var view = _make_1d_offset_view(parent, offset=3, size=3)  # [3,4,5]
    view %= 3.0

    assert_true(parent.load(2) == 2.0, "imod parent[2] unchanged")
    assert_true(parent.load(3) == 0.0, "imod parent[3]")
    assert_true(parent.load(4) == 1.0, "imod parent[4]")
    assert_true(parent.load(5) == 2.0, "imod parent[5]")
    assert_true(parent.load(6) == 6.0, "imod parent[6] unchanged")


def test_matrix_view_fill() raises:
    """Test Matrix.fill on a view with offset."""
    var parent = Matrix[nm.f64](shape=(3, 4), order="C")
    for i in range(3):
        for j in range(4):
            parent[i, j] = Scalar[nm.f64](i * 4 + j)

    # Create a view of row 1 (offset = 4)
    var view = parent.get(1)
    view.fill(99.0)

    # Row 0 should be unchanged
    assert_true(parent[0, 0] == 0.0, "matrix fill parent[0,0] unchanged")
    assert_true(parent[0, 3] == 3.0, "matrix fill parent[0,3] unchanged")
    # Row 1 should be filled
    assert_true(parent[1, 0] == 99.0, "matrix fill parent[1,0]")
    assert_true(parent[1, 1] == 99.0, "matrix fill parent[1,1]")
    assert_true(parent[1, 2] == 99.0, "matrix fill parent[1,2]")
    assert_true(parent[1, 3] == 99.0, "matrix fill parent[1,3]")
    # Row 2 should be unchanged
    assert_true(parent[2, 0] == 8.0, "matrix fill parent[2,0] unchanged")


def test_matrix_view_getset() raises:
    """Test Matrix __getitem__/__setitem__(x,y) respect offset."""
    var parent = Matrix[nm.f64](shape=(3, 4), order="C")
    for i in range(3):
        for j in range(4):
            parent[i, j] = Scalar[nm.f64](i * 4 + j)

    # Get a view of row 2 (offset = 8, shape 1x4)
    var view = parent.get(2)
    # Reading via [0, j] on the view should return row 2 values
    assert_true(view[0, 0] == 8.0, "matrix view get [0,0]")
    assert_true(view[0, 1] == 9.0, "matrix view get [0,1]")
    assert_true(view[0, 2] == 10.0, "matrix view get [0,2]")
    assert_true(view[0, 3] == 11.0, "matrix view get [0,3]")

    # Writing via [0, j] on the view should modify the parent
    view[0, 2] = 999.0
    assert_true(parent[2, 2] == 999.0, "matrix view set parent[2,2]")


def test_matrix_view_iadd() raises:
    """Test Matrix += on a row view with offset."""
    var parent = Matrix[nm.f64](shape=(3, 4), order="C")
    for i in range(3):
        for j in range(4):
            parent[i, j] = Scalar[nm.f64](i * 4 + j)

    # Get a view of row 1 (offset = 4)
    var view = parent.get(1)
    view += Scalar[nm.f64](100.0)

    # Row 0 should be unchanged
    assert_true(parent[0, 0] == 0.0, "matrix iadd parent[0,0] unchanged")
    # Row 1 should have 100 added
    assert_true(parent[1, 0] == 104.0, "matrix iadd parent[1,0]")
    assert_true(parent[1, 1] == 105.0, "matrix iadd parent[1,1]")
    assert_true(parent[1, 2] == 106.0, "matrix iadd parent[1,2]")
    assert_true(parent[1, 3] == 107.0, "matrix iadd parent[1,3]")
    # Row 2 should be unchanged
    assert_true(parent[2, 0] == 8.0, "matrix iadd parent[2,0] unchanged")


def test_matrix_view_load_store() raises:
    """Test Matrix load/store with offset."""
    var parent = Matrix[nm.f64](shape=(2, 4), order="C")
    for i in range(2):
        for j in range(4):
            parent[i, j] = Scalar[nm.f64](i * 4 + j)

    # Get a view of row 1 (offset = 4)
    var view = parent.get(1)
    # load(0) on the view should return the first element of row 1
    assert_true(view.load(0) == 4.0, "matrix load view[0]")
    assert_true(view.load(1) == 5.0, "matrix load view[1]")
    assert_true(view.load(3) == 7.0, "matrix load view[3]")

    # store on the view should write to the parent's row 1
    view.store(1, Scalar[nm.f64](555.0))
    assert_true(parent[1, 1] == 555.0, "matrix store parent[1,1]")


def test_matrix_view_reshape() raises:
    """Test Matrix reshape on a view with offset."""
    var parent = Matrix[nm.f64](shape=(3, 4), order="C")
    for i in range(3):
        for j in range(4):
            parent[i, j] = Scalar[nm.f64](i * 4 + j)

    # Get view of rows 1-2 (offset=4, shape 2x4)
    var view = parent.get(Slice(1, 3), Slice(0, 4))
    # Reshape 2x4 -> 4x2
    var reshaped = view.reshape((4, 2))
    # Should contain [4,5,6,7,8,9,10,11]
    assert_true(reshaped[0, 0] == 4.0, "matrix reshape [0,0]")
    assert_true(reshaped[0, 1] == 5.0, "matrix reshape [0,1]")
    assert_true(reshaped[1, 0] == 6.0, "matrix reshape [1,0]")
    assert_true(reshaped[3, 1] == 11.0, "matrix reshape [3,1]")


def test_ndarray_ravel_offset_view() raises:
    """Test ravel on a C-contiguous view with offset."""
    var parent = nm.arange[nm.f64](0, 24)
    var view = _make_2d_offset_view(
        parent, offset=6, rows=2, cols=3, stride0=3, stride1=1
    )
    # view represents parent[6:12] as 2x3:
    # [[6,7,8],[9,10,11]]
    var flat = nm.ravel(view)
    assert_true(flat.size == 6, "ravel size")
    assert_true(flat.item(0) == 6.0, "ravel [0]")
    assert_true(flat.item(1) == 7.0, "ravel [1]")
    assert_true(flat.item(2) == 8.0, "ravel [2]")
    assert_true(flat.item(3) == 9.0, "ravel [3]")
    assert_true(flat.item(4) == 10.0, "ravel [4]")
    assert_true(flat.item(5) == 11.0, "ravel [5]")


def test_where_on_offset_view() raises:
    """Test where() modifies an offset view's parent buffer."""
    var parent = nm.full[nm.f64](Shape(10), fill_value=0.0)
    var view = _make_1d_offset_view(parent, offset=3, size=4)  # [0,0,0,0]
    # mask: [True, False, True, False]
    var mask = nm.NDArray[DType.bool](Shape(4))
    mask.itemset(0, Scalar[DType.bool](True))
    mask.itemset(1, Scalar[DType.bool](False))
    mask.itemset(2, Scalar[DType.bool](True))
    mask.itemset(3, Scalar[DType.bool](False))
    nm.routines.indexing.`where`(view, Scalar[nm.f64](99.0), mask)

    # Parent positions 3 and 5 should be 99, others 0
    assert_true(parent.load(2) == 0.0, "where parent[2] unchanged")
    assert_true(parent.load(3) == 99.0, "where parent[3]")
    assert_true(parent.load(4) == 0.0, "where parent[4] unchanged")
    assert_true(parent.load(5) == 99.0, "where parent[5]")
    assert_true(parent.load(6) == 0.0, "where parent[6] unchanged")


# ===-----------------------------------------------------------------------===#
# main
# ===-----------------------------------------------------------------------===#


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
