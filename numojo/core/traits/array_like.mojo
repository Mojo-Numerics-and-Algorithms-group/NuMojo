# ===----------------------------------------------------------------------=== #
# NuMojo: Array-like trait
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Array-like (numojo.core.traits.array_like)
===========================================

Trait definitions for array-like behaviors.

Defines traits for array-like operations including loading/storing SIMD elements
and backend calculations (currently blocked by lack of trait parameterization).

Exports
-------
- Trait definitions for array-like types.
"""

# TODO: Implement once Mojo supports trait parameterization.

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray

# Blocked by lack of trait paramaterization

# trait Arraylike:
#     def load[width: Int](self, idx: Int) -> SIMD[dtype, width]:
#         """
#         Loads a SIMD element of size `width` at the given index `idx`.
#         """
#         ...
#     def store[width: Int](mut self, idx: Int, val: SIMD[dtype, width]):
#         """
#         Stores the SIMD element of size `width` at index `idx`.
#         """
#         ...

# trait NDArrayBackend:
#     """
#     A trait that defines backends for calculations in the rest of the library.
#     """

#     def __init__(mut self):
#         """
#         Initialize the backend.
#         """
#         ...

#     def math_func_1_array_in_one_array_out[
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

#     def math_func_2_array_in_one_array_out[
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

#     def math_func_one_array_one_SIMD_in_one_array_out[
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
