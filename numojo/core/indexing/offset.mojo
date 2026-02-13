# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Indexing offset calculation functions.
"""

from numojo.core.layout import NDArrayStrides
from numojo.core.indexing.item import Item


# TODO: Define a IndexContainerLike trait and use that to replace many of these get_1d_index overloads.
struct IndexMethods:
    @staticmethod
    fn get_1d_index(indices: List[Int], strides: NDArrayStrides) -> Int:
        """
        Get the flat index from a list of indices and NDArrayStrides.

        Args:
            indices: The list of indices.
            strides: The strides of the array.

        Returns:
            The flat index.
        """
        var idx: Int = 0
        for i in range(strides.ndim):
            idx += indices[i] * Int(strides.unsafe_load(i))
        return idx

    @staticmethod
    fn get_1d_index(indices: Item, strides: NDArrayStrides) -> Int:
        """
        Get the flat index from an Item and NDArrayStrides.

        Args:
            indices: The Item containing indices.
            strides: The strides of the array.

        Returns:
            The flat index.
        """
        var idx: Int = 0
        for i in range(strides.ndim):
            idx += Int(indices.unsafe_load(i) * strides.unsafe_load(i))
        return idx

    @staticmethod
    fn get_1d_index(indices: VariadicList[Int], strides: NDArrayStrides) -> Int:
        """
        Get the flat index from a variadic list of indices and NDArrayStrides.

        Args:
            indices: The variadic list of indices.
            strides: The strides of the array.

        Returns:
            The flat index.
        """
        var idx: Int = 0
        for i in range(strides.ndim):
            idx += indices[i] * Int(strides.unsafe_load(i))
        return idx

    @staticmethod
    fn get_1d_index(indices: List[Int], strides: List[Int]) -> Int:
        """
        Get the flat index from a list of indices and a list of strides.

        Args:
            indices: The list of indices.
            strides: The list of strides.

        Returns:
            The flat index.
        """
        var idx: Int = 0
        for i in range(len(strides)):
            idx += indices[i] * strides[i]
        return idx

    @staticmethod
    fn get_1d_index(
        indices: VariadicList[Int], strides: VariadicList[Int]
    ) -> Int:
        """
        Get the flat index from variadic lists of indices and strides.

        Args:
            indices: The variadic list of indices.
            strides: The variadic list of strides.

        Returns:
            The flat index.
        """
        var idx: Int = 0
        for i in range(len(strides)):
            idx += indices[i] * strides[i]
        return idx

    @staticmethod
    fn get_1d_index(indices: Tuple[Int, Int], strides: Tuple[Int, Int]) -> Int:
        """
        Get the flat index for a 2D matrix from tuples of indices and strides.

        Args:
            indices: The tuple of indices (row, col).
            strides: The tuple of strides.

        Returns:
            The flat index.
        """
        return indices[0] * strides[0] + indices[1] * strides[1]

    @staticmethod
    fn transfer_offset(offset: Int, strides: NDArrayStrides) raises -> Int:
        """
        Transfers the offset by flipping the strides information.
        Used to transfer between C-contiguous and F-continuous memory layouts.

        Args:
            offset: The offset in memory of an element.
            strides: The strides of the array.

        Returns:
            The offset in the flipped memory layout.
        """
        var remainder: Int = offset
        var indices: Item = Item(ndim=len(strides))
        for i in range(len(strides)):
            indices[i] = remainder // strides[i]
            remainder %= strides[i]

        return Self.get_1d_index(indices, strides.flipped())
