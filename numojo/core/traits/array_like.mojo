from numojo.core.ndarray import NDArray

# Blocked by lack of trait paramaterization

# trait Arraylike:
#     fn load[width: Int](self, idx: Int) -> SIMD[dtype, width]:
#         """
#         Loads a SIMD element of size `width` at the given index `idx`.
#         """
#         ...
#     fn store[width: Int](mut self, idx: Int, val: SIMD[dtype, width]):
#         """
#         Stores the SIMD element of size `width` at index `idx`.
#         """
#         ...

# trait NDArrayBackend:
#     """
#     A trait that defines backends for calculations in the rest of the library.
#     """

#     fn __init__(mut self):
#         """
#         Initialize the backend.
#         """
#         ...

#     fn math_func_1_array_in_one_array_out[
#         dtype: DType,
#         func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
#             type, simd_w
#         ],
#     ](self, array: Arraylike) -> Arraylike:
#         """
#         Apply a SIMD function of one variable and one return to a NDArray

#         Parameters:
#             dtype: The element type.
#             func: the SIMD function to to apply.

#         Args:
#             array: A NDArray

#         Returns:
#             A a new NDArray that is NDArray with the function func applied.
#         """
#         ...

#     fn math_func_2_array_in_one_array_out[
#         dtype: DType,
#         func: fn[type: DType, simd_w: Int] (
#             SIMD[type, simd_w], SIMD[type, simd_w]
#         ) -> SIMD[type, simd_w],
#     ](
#         self, array1: Arraylike, array2: Arraylike
#     ) raises -> Arraylike:
#         """
#         Apply a SIMD function of two variable and one return to a NDArray

#         Constraints:
#             Both arrays must have the same shape

#         Parameters:
#             dtype: The element type.
#             func: the SIMD function to to apply.

#         Args:
#             array1: A NDArray
#             array2: A NDArray

#         Returns:
#             A a new NDArray that is NDArray with the function func applied.
#         """

#         ...

#     fn math_func_one_array_one_SIMD_in_one_array_out[
#         dtype: DType,
#         func: fn[type: DType, simd_w: Int] (
#             SIMD[type, simd_w], SIMD[type, simd_w]
#         ) -> SIMD[type, simd_w],
#     ](
#         self, array: Arraylike, scalar: Scalar[dtype]
#     ) -> Arraylike:
#         """
#         Apply a SIMD function of two variable and one return to a NDArray

#         Constraints:
#             Both arrays must have the same shape

#         Parameters:
#             dtype: The element type.
#             func: the SIMD function to to apply.

#         Args:
#             array: A NDArray
#             scalar: A Scalar

#         Returns:
#             A a new NDArray that is NDArray with the function func applied.
#         """

#         ...
