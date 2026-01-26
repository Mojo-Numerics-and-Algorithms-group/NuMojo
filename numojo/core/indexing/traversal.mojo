# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Functions to traverse a multi-dimensional array.
"""

from memory import UnsafePointer
from numojo.core.layout import NDArrayShape, NDArrayStrides
from numojo.core.indexing.offset import IndexMethods
from numojo.core.error import NumojoError


struct TraverseMethods:
    @staticmethod
    fn traverse_buffer_according_to_shape_and_strides[
        origin: MutOrigin
    ](
        mut ptr: UnsafePointer[Scalar[DType.int], origin=origin],
        shape: NDArrayShape,
        strides: NDArrayStrides,
        current_dim: Int = 0,
        previous_sum: Int = 0,
    ) raises:
        """
        Store sequence of indices according to shape and strides into the pointer.
        Auxiliary function for variadic number of dimensions.

        UNSAFE: Raw pointer is used!

        Parameters:
            origin: The mutability origin of the pointer.

        Args:
            ptr: Pointer to buffer of uninitialized 1-d index array.
            shape: The shape of the array.
            strides: The strides of the array.
            current_dim: Temporarily save the current dimension.
            previous_sum: Temporarily save the previous summed index.
        """
        for index_of_axis in range(Int(shape[current_dim])):
            var current_sum = previous_sum + index_of_axis * Int(
                strides[current_dim]
            )
            if current_dim >= shape.ndim - 1:
                ptr.init_pointee_copy(current_sum)
                ptr += 1
            else:
                Self.traverse_buffer_according_to_shape_and_strides(
                    ptr,
                    shape,
                    strides,
                    current_dim + 1,
                    current_sum,
                )

    @staticmethod
    fn traverse_iterative[
        dtype: DType
    ](
        orig: NDArray[dtype],
        mut narr: NDArray[dtype],
        ndim: List[Int],
        coefficients: List[Int],
        strides: List[Int],
        offset: Int,
        mut index: List[Int],
        depth: Int,
    ) raises:
        """
        Traverse a multi-dimensional array in an iterative manner.

        Parameters:
            dtype: The data type of the NDArray elements.

        Args:
            orig: The original array.
            narr: The array to store the result.
            ndim: The number of dimensions of the array.
            coefficients: The coefficients to traverse the sliced part of the original array.
            strides: The strides to traverse the new NDArray `narr`.
            offset: The offset to the first element of the original NDArray.
            index: The list of indices.
            depth: The depth of the indices.
        """
        var total_elements = narr.size

        for _ in range(total_elements):
            var orig_idx = offset
            for i in range(len(index)):
                orig_idx += index[i] * coefficients[i]

            var narr_idx = 0
            for i in range(len(index)):
                narr_idx += index[i] * strides[i]

            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")

            narr._buf.ptr.store(narr_idx, orig._buf.ptr.load[width=1](orig_idx))

            for d in range(ndim.__len__() - 1, -1, -1):
                index[d] += 1
                if index[d] < ndim[d]:
                    break
                index[d] = 0

    @staticmethod
    fn traverse_iterative_setter[
        dtype: DType
    ](
        orig: NDArray[dtype],
        mut narr: NDArray[dtype],
        ndim: List[Int],
        coefficients: List[Int],
        strides: List[Int],
        offset: Int,
        mut index: List[Int],
    ) raises:
        """
        Traverse a multi-dimensional array in an iterative manner for setter.

        Parameters:
            dtype: The data type of the NDArray elements.

        Args:
            orig: The original array (source).
            narr: The array to store the result (destination).
            ndim: The number of dimensions of the array.
            coefficients: The coefficients to traverse the sliced part of the original array.
            strides: The strides to traverse the new NDArray `narr`.
            offset: The offset to the first element of the original NDArray.
            index: The list of indices.
        """
        var total_elements = narr.size

        for _ in range(total_elements):
            var orig_idx = offset
            for i in range(len(index)):
                orig_idx += index[i] * coefficients[i]

            var narr_idx = 0
            for i in range(len(index)):
                narr_idx += index[i] * strides[i]

            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")

            narr._buf.ptr.store(orig_idx, orig._buf.ptr.load[width=1](narr_idx))

            for d in range(ndim.__len__() - 1, -1, -1):
                index[d] += 1
                if index[d] < ndim[d]:
                    break
                index[d] = 0
