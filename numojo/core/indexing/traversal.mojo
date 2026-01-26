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

struct TraverseMethods:
    @staticmethod
    fn traverse_buffer_according_to_shape_and_strides[
        origin: ImmutOrigin
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
            var current_sum = (
                previous_sum + index_of_axis * Int(strides[current_dim])
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
        dtype: DType,
        src_origin: ImmutOrigin,
        dest_origin: ImmutOrigin,
    ](
        src_ptr: UnsafePointer[Scalar[dtype], origin=src_origin],
        dest_ptr: UnsafePointer[Scalar[dtype], origin=dest_origin],
        shape: List[Int],
        coefficients: List[Int],
        offset: Int,
        total_elements: Int,
    ) raises:
        """
        Traverse a multi-dimensional source and copy to a contiguous destination.

        Parameters:
            dtype: The data type of the elements.
            src_origin: Origin of the source pointer.
            dest_origin: Origin of the destination pointer.

        Args:
            src_ptr: Source pointer.
            dest_ptr: Destination pointer (contiguous).
            shape: The logical shape of the source view.
            coefficients: The coefficients (strides) of the source view.
            offset: The base offset of the source view.
            total_elements: Total number of elements to copy.
        """
        var index = List[Int]()
        for _ in range(len(shape)):
            index.append(0)

        for lin in range(total_elements):
            var src_idx = offset + IndexMethods.get_1d_index(index, coefficients)
            dest_ptr.store(lin, src_ptr.load[width=1](src_idx))

            for d in range(len(shape) - 1, -1, -1):
                index[d] += 1
                if index[d] < shape[d]:
                    break
                index[d] = 0

    @staticmethod
    fn traverse_iterative_setter[
        dtype: DType,
        src_origin: ImmutOrigin,
        dest_origin: ImmutOrigin,
    ](
        src_ptr: UnsafePointer[Scalar[dtype], origin=src_origin],
        dest_ptr: UnsafePointer[Scalar[dtype], origin=dest_origin],
        shape: List[Int],
        coefficients: List[Int],
        offset: Int,
        total_elements: Int,
    ) raises:
        """
        Traverse a contiguous source and set into a multi-dimensional destination view.

        Parameters:
            dtype: The data type of the elements.
            src_origin: Origin of the source pointer (contiguous).
            dest_origin: Origin of the destination pointer (view).

        Args:
            src_ptr: Source pointer (contiguous).
            dest_ptr: Destination pointer (view).
            shape: The logical shape of the destination view.
            coefficients: The coefficients (strides) of the destination view.
            offset: The base offset of the destination view.
            total_elements: Total number of elements to copy.
        """
        var index = List[Int]()
        for _ in range(len(shape)):
            index.append(0)

        for lin in range(total_elements):
            var dest_idx = offset + IndexMethods.get_1d_index(index, coefficients)
            dest_ptr.store(dest_idx, src_ptr.load[width=1](lin))

            for d in range(len(shape) - 1, -1, -1):
                index[d] += 1
                if index[d] < shape[d]:
                    break
                index[d] = 0
