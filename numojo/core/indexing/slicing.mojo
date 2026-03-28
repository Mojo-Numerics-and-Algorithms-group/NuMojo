# ===----------------------------------------------------------------------=== #
# NuMojo: Slicing
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Slicing (numojo.core.indexing.slicing)

This module defines internal data structures and utilities for handling slicing operations in NuMojo.
"""

from std.math import ceil


# ===----------------------------------------------------------------------=== #
# Internal Data Structure: IndexTypeInfo
# ===----------------------------------------------------------------------=== #
struct IndexTypeInfo(ImplicitlyCopyable):
    var is_integer: Bool
    var is_slice: Bool
    var is_ellipsis: Bool
    var is_newaxis: Bool

    def __init__(
        out self,
        is_integer: Bool = False,
        is_slice: Bool = False,
        is_ellipsis: Bool = False,
        is_newaxis: Bool = False,
    ):
        self.is_integer = is_integer
        self.is_slice = is_slice
        self.is_ellipsis = is_ellipsis
        self.is_newaxis = is_newaxis

    def __repr__(self) -> String:
        return (
            "IndexTypeInfo(is_integer={}, is_slice={}, is_ellipsis={},"
            " is_newaxis={})".format(
                self.is_integer,
                self.is_slice,
                self.is_ellipsis,
                self.is_newaxis,
            )
        )

    def __str__(self) -> String:
        return (
            "IndexTypeInfo(is_integer={}, is_slice={}, is_ellipsis={},"
            " is_newaxis={})".format(
                self.is_integer,
                self.is_slice,
                self.is_ellipsis,
                self.is_newaxis,
            )
        )

    def size(self) -> Int:
        """
        Returns the number of active index types in this IndexTypeInfo.
        """
        var size = 0
        if self.is_integer:
            size += 1
        if self.is_slice:
            size += 1
        if self.is_ellipsis:
            size += 1
        if self.is_newaxis:
            size += 1
        return size


# ===----------------------------------------------------------------------=== #
# Internal Data Structure: InternalSlice
# ===----------------------------------------------------------------------=== #
struct InternalSlice(ImplicitlyCopyable):
    var start: Int
    var end: Int
    var step: Int

    def __init__(out self, start: Int, end: Int, step: Int):
        self.start = start
        self.end = end
        self.step = step

    def __repr__(self) -> String:
        return "InternalSlice(start={}, end={}, step={})".format(
            self.start, self.end, self.step
        )

    def __str__(self) -> String:
        return "InternalSlice(start={}, end={}, step={})".format(
            self.start, self.end, self.step
        )

    def __eq__(self, other: Self) -> Bool:
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    def __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    def to_tuple(self) -> Tuple[Int, Int, Int]:
        return (self.start, self.end, self.step)

    def to_slice(self) -> Slice:
        return Slice(self.start, self.end, self.step)

    def normalize(self, dim: Int) -> InternalSlice:
        var start_norm = self.start
        var end_norm = self.end

        if self.start < 0:
            start_norm = dim + self.start
        if self.end < 0:
            end_norm = dim + self.end

        return InternalSlice(start_norm, end_norm, self.step)

    def check_bounds(self, dim: Int) raises:
        if self.start < 0 or self.start >= dim:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Slice start index {} out of bounds for dimension of"
                        " size {}".format(self.start, dim)
                    ),
                    location="InternalSlice.check_bounds()",
                )
            )
        if self.end < 0 or self.end > dim:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Slice end index {} out of bounds for dimension of"
                        " size {}".format(self.end, dim)
                    ),
                    location="InternalSlice.check_bounds()",
                )
            )
        if self.step == 0:
            raise Error(
                NumojoError(
                    category="index",
                    message="Slice step cannot be zero",
                    location="InternalSlice.check_bounds()",
                )
            )

    @staticmethod
    def get_slice_info(s: Slice, dim: Int) -> Tuple[Int, Int, Int, Int]:
        """
        Get complete slice information for a given dimension.

        Args:
            s: The slice to process.
            dim: The dimension size to process against.

        Returns:
            A tuple of (start, end, step, length) for the slice.

        Notes:
            For cases with step = 0, error handling should be done prior to calling this function.
        """
        var start: Int
        var end: Int
        var step: Int = s.step.or_else(1)
        var length: Int

        if step > 0:
            start = s.start.or_else(0)
            end = s.end.or_else(dim)
        else:
            start = s.start.or_else(dim - 1)
            end = s.end.or_else(-dim - 1)

        if start < 0:
            start += dim
        if end < 0:
            end += dim

        if step > 0:
            length = max((end - start + step - 1) // step, 0)
        else:
            length = max((start - end - step - 1) // -step, 0)

        return (start, end, step, length)

    def get_slice_info(self, dim: Int) -> Tuple[Int, Int, Int, Int]:
        """
        Get complete slice information for a given dimension.

        Args:
            dim: The dimension size to process against.

        Returns:
            A tuple of (start, end, step, length) for the slice.
        """
        var length = Int(ceil((self.end - self.start) / self.step))
        return (self.start, self.end, self.step, length)
