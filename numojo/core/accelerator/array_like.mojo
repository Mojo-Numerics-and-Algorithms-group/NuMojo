# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator array access traits
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator array access traits (numojo.core.accelerator.array_like).
=====================================================================
Origin-preserving pointer access shared by accelerator-array owners and views.

Exports
-------
- `NDArrayLike`: Common read-only interface for owners and views.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.device import Device
from numojo.core.layout.ndlayout import NDLayout, Shape, Strides


trait NDArrayLike(Sized):
    """Common read-only interface implemented by array owners and views."""

    comptime ElementType: DType
    comptime DeviceType: Device
    comptime Writable: Bool

    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied shape, strides, offset, and size metadata.

        Examples:
            Use `array.layout()` in generic indexing code.
        """
        ...

    def ndim(self) -> Int:
        """Return the logical rank.

        Returns:
            Number of array dimensions.

        Examples:
            Use `array.ndim()` in a generic rank check.
        """
        ...

    def shape(self) -> Shape:
        """Return a copy of the shape.

        Returns:
            Copied logical dimensions.

        Examples:
            Use `array.shape()` when allocating an output owner.
        """
        ...

    def strides(self) -> Strides:
        """Return a copy of the element strides.

        Returns:
            Copied logical strides.

        Examples:
            Use `array.strides()` in generic traversal code.
        """
        ...

    def size(self) -> Int:
        """Return the logical element count.

        Returns:
            Product of the shape dimensions.

        Examples:
            Use `array.size()` to bound a flat traversal.
        """
        ...

    def offset(self) -> Int:
        """Return the first logical element's storage offset.

        Returns:
            Element offset within the backing allocation.

        Examples:
            Use `array.offset()` when adapting a layout.
        """
        ...

    def is_c_contiguous(self) -> Bool:
        """Return whether the logical layout is C contiguous.

        Returns:
            Whether flat C-order pointer traversal is valid.

        Examples:
            Check `array.is_c_contiguous()` before a SIMD pointer loop.
        """
        ...

    def is_f_contiguous(self) -> Bool:
        """Return whether the logical layout is Fortran contiguous.

        Returns:
            Whether flat Fortran-order traversal is valid.

        Examples:
            Use `array.is_f_contiguous()` to select a traversal order.
        """
        ...

    def owns_data(self) -> Bool:
        """Return whether this value owns its backing allocation.

        Returns:
            `True` for an owning array and `False` for a view.

        Examples:
            Use `array.owns_data()` for diagnostics.
        """
        ...

    def is_writeable(self) -> Bool:
        """Return whether this value exposes mutable storage.

        Returns:
            Runtime mirror of the value's write capability.

        Examples:
            Use `array.is_writeable()` for diagnostics and validation.
        """
        ...

    def is_cpu(self) -> Bool:
        """Return whether storage is host-addressable CPU memory.

        Returns:
            `True` for `Device.CPU`.

        Examples:
            Use `array.is_cpu()` before host pointer access.
        """
        ...

    def is_gpu(self) -> Bool:
        """Return whether storage is accelerator memory.

        Returns:
            `True` for a GPU device.

        Examples:
            Use `array.is_gpu()` before launching a kernel.
        """
        ...

    def num_elements(self) -> Int:
        """Return the logical element count.

        Returns:
            Same value as `size()`.

        Examples:
            Use `array.num_elements()` in MAX-compatible adapters.
        """
        ...

    def item(self, flat_index: Int) raises -> Scalar[Self.ElementType]:
        """Return one element at a logical flat index.

        Args:
            flat_index: C-order logical flat index.

        Returns:
            Selected scalar value.

        Raises:
            NumojoError: If the index is invalid or storage is not host-accessible.

        Examples:
            Use `array.item(0)` to read the first logical element.
        """
        ...

    def item(self, *indices: Int) raises -> Scalar[Self.ElementType]:
        """Return one element at N-dimensional coordinates.

        Args:
            indices: One coordinate per axis.

        Returns:
            Selected scalar value.

        Raises:
            NumojoError: If coordinates are invalid or storage is not host-accessible.

        Examples:
            Use `array.item(1, 2)` to read a matrix element.
        """
        ...

    def unsafe_ptr[
        o: Origin
    ](ref[o] self,) -> Pointer[Scalar[Self.ElementType], ImmOrigin(o)]:
        """Return an immutable pointer tied to the current borrow.

        Parameters:
            o: Origin of the current array or view borrow.

        Returns:
            Immutable pointer whose origin keeps the array or view alive.

        Examples:
            Use `array.unsafe_ptr()` in generic read-only algorithms.
        """
        ...

    def unsafe_mut_ptr[
        o: MutOrigin
    ](ref[o] self,) -> Pointer[Scalar[Self.ElementType], o]:
        """Return a mutable pointer tied to the current mutable borrow.

        Parameters:
            o: Mutable origin of the current array or view borrow.

        Returns:
            Mutable pointer whose origin keeps the array or view alive.

        Examples:
            Use `array.unsafe_mut_ptr()` in generic output algorithms.
        """
        ...
