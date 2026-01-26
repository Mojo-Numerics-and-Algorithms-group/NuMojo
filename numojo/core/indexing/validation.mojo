# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Validation module for indexing.
"""

from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape, NDArrayStrides


struct Validator:
    @staticmethod
    fn normalize(index: Int, dim: Int) -> Int:
        """
        Normalize a possibly negative index.

        Args:
            index: The index to normalize.
            dim: The size of the dimension.

        Returns:
            The normalized index.
        """
        return index if index >= 0 else index + dim

    @staticmethod
    fn check_bounds(index: Int, dim: Int, axis: Int = 0) raises:
        """
        Check if an index is within bounds for a dimension.

        Args:
            index: The index to check.
            dim: The size of the dimension.
            axis: The axis index (for error reporting).

        Raises:
            Error: If the index is out of bounds.
        """
        if index >= dim or index < -dim:
            raise Error(
                NumojoError(
                    category="index",
                    message="Index "
                    + String(index)
                    + " out of bounds for axis "
                    + String(axis)
                    + " with size "
                    + String(dim),
                    location="Validator.check_bounds",
                )
            )

    @staticmethod
    fn validate_reshape(current_size: Int, new_shape: NDArrayShape) raises:
        """
        Validate if a reshape operation is valid.

        Args:
            current_size: Current total number of elements.
            new_shape: The target shape.

        Raises:
            Error: If the reshape is invalid.
        """
        if current_size != new_shape.size():
            raise Error(
                NumojoError(
                    category="shape",
                    message="Cannot reshape array of size "
                    + String(current_size)
                    + " into shape "
                    + String(new_shape),
                    location="Validator.validate_reshape",
                )
            )

    @staticmethod
    fn validate_and_normalize_axes(
        rank: Int, axes: List[Int]
    ) raises -> List[Int]:
        """
        Validate and normalize axes for reduction operations.

        Args:
            rank: The rank of the array.
            axes: The input axes.

        Returns:
            Normalized axes.

        Raises:
            Error: If any axis is invalid or duplicated.
        """
        var normalized = List[Int]()
        if len(axes) == 0:
            for i in range(rank):
                normalized.append(i)
            return normalized^

        var seen = List[Bool]()
        for _ in range(rank):
            seen.append(False)

        for axis in axes:
            var a = axis if axis >= 0 else axis + rank
            if a < 0 or a >= rank:
                raise Error(
                    NumojoError(
                        category="index",
                        message="Axis " + String(axis) + " out of range",
                        location="Validator.validate_and_normalize_axes",
                    )
                )
            if seen[a]:
                raise Error(
                    NumojoError(
                        category="index",
                        message="Duplicate axis "
                        + String(axis)
                        + " in reduction",
                        location="Validator.validate_and_normalize_axes",
                    )
                )
            seen[a] = True
            normalized.append(a)

        return normalized^
