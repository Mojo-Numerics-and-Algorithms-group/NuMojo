"""
Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
"""
# ===----------------------------------------------------------------------=== #
# Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
# Last updated: 2024-09-08
# ===----------------------------------------------------------------------=== #

from algorithm.functional import vectorize

from python import Python, PythonObject
from .ndarray import NDArray, NDArrayShape, NDArrayStride


@value
struct _IdxIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for idx.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying idx data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Idx
    var length: Int

    fn __init__(
        inout self,
        array: Idx,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) raises -> Scalar[DType.index]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.array.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.array.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index


struct Idx(CollectionElement, Formattable):
    alias dtype: DType = DType.index
    alias width = simdwidthof[Self.dtype]()
    var storage: UnsafePointer[Scalar[Self.dtype]]
    var len: Int

    @always_inline("nodebug")
    fn __init__(inout self, owned *args: Scalar[Self.dtype]):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(args.__len__())
        self.len = args.__len__()
        for i in range(args.__len__()):
            self.storage[i] = args[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, owned args: Variant[List[Int], VariadicList[Int]]
    ) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        if args.isa[List[Int]]():
            self.len = args[List[Int]].__len__()
            self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(self.len)
            for i in range(self.len):
                self.storage[i] = args[List[Int]][i]
        elif args.isa[VariadicList[Int]]():
            self.len = args[VariadicList[Int]].__len__()
            self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(self.len)
            for i in range(self.len):
                self.storage[i] = args[VariadicList[Int]][i]
        else:
            raise Error("Invalid type")

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.storage = UnsafePointer[Scalar[Self.dtype]]().alloc(
            other.__len__()
        )
        self.len = other.len
        for i in range(other.__len__()):
            self.storage[i] = other[i]

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Self):
        """Move construct the tuple.

        Args:
            other: The tuple to move.
        """
        self.storage = other.storage
        self.len = other.len

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.len

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        """Get the value at the specified index.

        Args:
            index: The index of the value to get.

        Returns:
            The value at the specified index.
        """
        return int(self.storage[index])

    @always_inline("nodebug")
    fn __setitem__(self, index: Int, val: Scalar[Self.dtype]):
        """Set the value at the specified index.

        Args:
            index: The index of the value to set.
            val: The value to set.
        """
        self.storage[index] = val

    fn __iter__(self) raises -> _IdxIter[__lifetime_of(self)]:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _IdxIter[__lifetime_of(self)](
            array=self,
            length=self.len,
        )

    fn format_to(self, inout writer: Formatter):
        writer.write("Idx: " + self.str() + "\n" + "Length: " + str(self.len))

    fn str(self) -> String:
        var result: String = "["
        for i in range(self.len):
            result += str(self.storage[i])
            if i < self.len - 1:
                result += ", "
        result += "]"
        return result


fn fill_pointer[
    dtype: DType
](
    inout array: UnsafePointer[Scalar[dtype]], size: Int, value: Scalar[dtype]
) raises:
    """
    Fill a NDArray with a specific value.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        array: The pointer to the NDArray.
        size: The size of the NDArray.
        value: The value to fill the NDArray with.
    """
    alias width = simdwidthof[dtype]()

    @parameter
    fn vectorized_fill[simd_width: Int](idx: Int):
        array.store[width=simd_width](idx, value)

    vectorize[vectorized_fill, width](size)


# define a ndarray internal trait and remove multiple overloads of these _get_index
fn _get_index(indices: List[Int], weights: NDArrayShape) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndlen):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: NDArrayShape) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndlen):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: NDArrayStride) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndlen):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: Idx, weights: NDArrayStride) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var index: Int = 0
    for i in range(weights.ndlen):
        index += indices[i] * weights[i]
    return index


fn _get_index(indices: VariadicList[Int], weights: NDArrayStride) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndlen):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: List[Int]) -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: VariadicList[Int]) -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _traverse_iterative[
    dtype: DType
](
    orig: NDArray[dtype],
    inout narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    inout index: List[Int],
    depth: Int,
) raises:
    """
    Traverse a multi-dimensional array in a iterative manner.

    Raises:
        Error: If the index is out of bound.

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
    var total_elements = narr.ndshape.ndsize

    # # parallelized version was slower xD
    for _ in range(total_elements):
        var orig_idx = offset + _get_index(index, coefficients)
        var narr_idx = _get_index(index, strides)
        try:
            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")
        except:
            return

        narr.data.store[width=1](narr_idx, orig.data.load[width=1](orig_idx))

        for d in range(ndim.__len__() - 1, -1, -1):
            index[d] += 1
            if index[d] < ndim[d]:
                break
            index[d] = 0


fn _traverse_iterative_setter[
    dtype: DType
](
    orig: NDArray[dtype],
    inout narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    inout index: List[Int],
) raises:
    """
    Traverse a multi-dimensional array in a iterative manner.

    Raises:
        Error: If the index is out of bound.

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
    """
    var total_elements = narr.ndshape.ndsize

    for _ in range(total_elements):
        var orig_idx = offset + _get_index(index, coefficients)
        var narr_idx = _get_index(index, strides)
        try:
            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")
        except:
            return

        narr.data.store[width=1](orig_idx, orig.data.load[width=1](narr_idx))

        for d in range(ndim.__len__() - 1, -1, -1):
            index[d] += 1
            if index[d] < ndim[d]:
                break
            index[d] = 0


fn bool_to_numeric[
    dtype: DType
](array: NDArray[DType.bool]) raises -> NDArray[dtype]:
    """
    Convert a boolean NDArray to a numeric NDArray.

    Parameters:
        dtype: The data type of the output NDArray elements.

    Args:
        array: The boolean NDArray to convert.

    Returns:
        The converted NDArray of type `dtype` with 1s (True) and 0s (False).
    """
    # Can't use simd becuase of bit packing error
    var res: NDArray[dtype] = NDArray[dtype](array.shape())
    for i in range(array.size()):
        var t: Bool = array.item(i)
        if t:
            res.data[i] = 1
        else:
            res.data[i] = 0
    return res


fn to_numpy[dtype: DType](array: NDArray[dtype]) raises -> PythonObject:
    """
    Convert a NDArray to a numpy array.

    Example:
    ```console
    var arr = NDArray[DType.float32](3, 3, 3)
    var np_arr = to_numpy(arr)
    var np_arr1 = arr.to_numpy()
    ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        array: The NDArray to convert.

    Returns:
        The converted numpy array.
    """
    try:
        var np = Python.import_module("numpy")

        np.set_printoptions(4)

        var dimension = array.ndim
        var np_arr_dim = PythonObject([])

        for i in range(dimension):
            np_arr_dim.append(array.ndshape[i])

        # Implement a dictionary for this later
        var numpyarray: PythonObject
        var np_dtype = np.float64
        if dtype == DType.float16:
            np_dtype = np.float16
        elif dtype == DType.float32:
            np_dtype = np.float32
        elif dtype == DType.int64:
            np_dtype = np.int64
        elif dtype == DType.int32:
            np_dtype = np.int32
        elif dtype == DType.int16:
            np_dtype = np.int16
        elif dtype == DType.int8:
            np_dtype = np.int8

        numpyarray = np.empty(np_arr_dim, dtype=np_dtype)
        var pointer_d = numpyarray.__array_interface__["data"][
            0
        ].unsafe_get_as_pointer[dtype]()
        memcpy(pointer_d, array.unsafe_ptr(), array.num_elements())
        _ = array

        return numpyarray^

    except e:
        print("Error in converting to numpy", e)
        return PythonObject()
