"""
Implements generic reusable functions for math
"""
from tensor import Tensor
from testing import assert_raises
from algorithm.functional import parallelize, vectorize, num_physical_cores

trait Backend:
    """
    A trait that defines backends for calculations in the rest of the library.
    """

    fn __init__(inout self:Self):
       pass

    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[dtype]:
        """
        Apply a SIMD function of one variable and one return to a tensor

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        ...


    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
        """
        Apply a SIMD function of two variable and one return to a tensor

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor1: A tensor
            tensor2: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """

        ...

    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
       ...

    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[DType.bool]:
        ...


    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor1: Tensor[dtype], intval: Int) -> Tensor[dtype]:
        ...

struct VectorizedVerbose(Backend):
    """
    Vectorized Backend Struct.

    Defualt Numojo computation backend takes advantage of SIMD.
    Uses defualt simdwidth.
    """
    
    fn __init__(inout self:Self):
       pass


    
    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self: Self, tensor: Tensor[dtype]) -> Tensor[dtype]:
        """
        Apply a SIMD function of one variable and one return to a tensor

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (tensor.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data = tensor.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        if tensor.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (tensor.num_elements() // opt_nelts),
                tensor.num_elements(),
            ):
                var simd_data = func[dtype, 1](tensor.load[width=1](i))
                result_tensor.store[width=1](i, simd_data)
        return result_tensor


    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
        """
        Apply a SIMD function of two variable and one return to a tensor

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor1: A tensor
            tensor2: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """

        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (tensor1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = tensor1.load[width=opt_nelts](i)
            var simd_data2 = tensor2.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        if tensor1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (tensor1.num_elements() // opt_nelts),
                tensor1.num_elements(),
            ):
                var simd_data1 = tensor1.load[width=1](i)
                var simd_data2 = tensor2.load[width=1](i)
                result_tensor.store[width=1](
                    i, func[dtype, 1](simd_data1, simd_data2)
                )
        return result_tensor


    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (tensor1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = tensor1.load[width=opt_nelts](i)
            var simd_data2 = tensor2.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        if tensor1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (tensor1.num_elements() // opt_nelts),
                tensor1.num_elements(),
            ):
                var simd_data1 = tensor1.load[width=1](i)
                var simd_data2 = tensor2.load[width=1](i)
                result_tensor.store[width=1](
                    i, func[dtype, 1](simd_data1, simd_data2)
                )
        return result_tensor


    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[DType.bool]:
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (tensor.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data = tensor.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        if tensor.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (tensor.num_elements() // opt_nelts),
                tensor.num_elements(),
            ):
                var simd_data = func[dtype, 1](tensor.load[width=1](i))
                result_tensor.store[width=1](i, simd_data)
        return result_tensor


    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor1: Tensor[dtype], intval: Int) -> Tensor[dtype]:
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        for i in range(
            0, opt_nelts * (tensor1.num_elements() // opt_nelts), opt_nelts
        ):
            var simd_data1 = tensor1.load[width=opt_nelts](i)

            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, intval)
            )

        if tensor1.num_elements() % opt_nelts != 0:
            for i in range(
                opt_nelts * (tensor1.num_elements() // opt_nelts),
                tensor1.num_elements(),
            ):
                var simd_data1 = tensor1.load[width=1](i)
                result_tensor.store[width=1](i, func[dtype, 1](simd_data1, intval))
        return result_tensor


struct Vectorized(Backend):
    """
    Vectorized Backend Struct.

    Defualt Numojo computation backend takes advantage of SIMD.
    Uses defualt simdwidth.
    """
    
    fn __init__(inout self:Self):
       pass


    
    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self: Self, tensor: Tensor[dtype]) -> Tensor[dtype]:
        """
        Apply a SIMD function of one variable and one return to a tensor

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
       
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = tensor.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        vectorize[closure, opt_nelts](tensor.num_elements())
        
        return result_tensor


    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
        """
        Apply a SIMD function of two variable and one return to a tensor

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor1: A tensor
            tensor2: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """

        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = tensor1.load[width=opt_nelts](i)
            var simd_data2 = tensor2.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        vectorize[closure, opt_nelts](tensor1.num_elements())
        return result_tensor


    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
       
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = tensor1.load[width=opt_nelts](i)
            var simd_data2 = tensor2.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data1, simd_data2)
            )

        vectorize[closure, opt_nelts](tensor1.num_elements())
        return result_tensor
        


    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[DType.bool]:
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = tensor.load[width=opt_nelts](i)
            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data)
            )

        vectorize[closure, opt_nelts](tensor.num_elements())
        return result_tensor


    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype], intval: Int) -> Tensor[dtype]:

        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = tensor.load[width=opt_nelts](i)

            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data, intval)
            )

        vectorize[closure, opt_nelts](tensor.num_elements())
        return result_tensor


struct VectorizedParallelized(Backend):
    """
    Vectorized and parrallelized Backend Struct.

    Currently an order of magnitude slower than Vectorized for most functions.
    No idea why, Not Reccomened for use at this Time.
    """
    
    fn __init__(inout self:Self):
       pass


    
    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self: Self, tensor: Tensor[dtype]) -> Tensor[dtype]:
        """
        Apply a SIMD function of one variable and one return to a tensor

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = tensor.num_elements()//num_cores
        var comps_remainder: Int = tensor.num_elements()%num_cores
        var remainder_offset: Int = num_cores * comps_per_core
        @parameter
        fn par_closure(j:Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = tensor.load[width=opt_nelts](i+comps_per_core*j)
                result_tensor.store[width=opt_nelts](
                    i+comps_per_core*j, func[dtype, opt_nelts](simd_data)
                )

            vectorize[closure, opt_nelts](comps_per_core)
        parallelize[par_closure]()
        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = tensor.load[width=opt_nelts](i+remainder_offset)
            result_tensor.store[width=opt_nelts](
                i+remainder_offset, func[dtype, opt_nelts](simd_data)
            )
        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_tensor


    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
        """
        Apply a SIMD function of two variable and one return to a tensor

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor1: A tensor
            tensor2: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """

        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = tensor1.num_elements()//num_cores
        var comps_remainder: Int = tensor1.num_elements()%num_cores
        var remainder_offset: Int = num_cores * comps_per_core
        @parameter
        fn par_closure(j:Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = tensor1.load[width=opt_nelts](i+comps_per_core*j)
                var simd_data2 = tensor2.load[width=opt_nelts](i+comps_per_core*j)
                result_tensor.store[width=opt_nelts](
                    i+comps_per_core*j, func[dtype, opt_nelts](simd_data1, simd_data2)
                )

            vectorize[closure, opt_nelts](comps_per_core)
        parallelize[par_closure]()
        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = tensor1.load[width=opt_nelts](i+remainder_offset)
            var simd_data2 = tensor2.load[width=opt_nelts](i+remainder_offset)
            result_tensor.store[width=opt_nelts](
                i+remainder_offset, func[dtype, opt_nelts](simd_data1, simd_data2)
            )
        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_tensor


    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor1.shape())
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int = num_physical_cores()
        var comps_per_core: Int = tensor1.num_elements()//num_cores
        var comps_remainder: Int = tensor1.num_elements()%num_cores
        var remainder_offset: Int = num_cores * comps_per_core
        @parameter
        fn par_closure(j:Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data1 = tensor1.load[width=opt_nelts](i+comps_per_core*j)
                var simd_data2 = tensor2.load[width=opt_nelts](i+comps_per_core*j)
                result_tensor.store[width=opt_nelts](
                    i+comps_per_core*j, func[dtype, opt_nelts](simd_data1, simd_data2)
                )

            vectorize[closure, opt_nelts](comps_per_core)
        parallelize[par_closure]()
        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data1 = tensor1.load[width=opt_nelts](i+remainder_offset)
            var simd_data2 = tensor2.load[width=opt_nelts](i+remainder_offset)
            result_tensor.store[width=opt_nelts](
                i+remainder_offset, func[dtype, opt_nelts](simd_data1, simd_data2)
            )
        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_tensor
        


    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[DType.bool]:
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        var num_cores: Int =  num_physical_cores()
        var comps_per_core: Int = tensor.num_elements()//num_cores
        var comps_remainder: Int = tensor.num_elements()%num_cores
        var remainder_offset: Int = num_cores * comps_per_core
        @parameter
        fn par_closure(j:Int):
            @parameter
            fn closure[simdwidth: Int](i: Int):
                var simd_data = tensor.load[width=opt_nelts](i+comps_per_core*j)
                result_tensor.store[width=opt_nelts](
                    i+comps_per_core*j, func[dtype, opt_nelts](simd_data)
                )

            vectorize[closure, opt_nelts](comps_per_core)
        parallelize[par_closure]()
        @parameter
        fn remainder_closure[simdwidth: Int](i: Int):
            var simd_data = tensor.load[width=opt_nelts](i+remainder_offset)
            result_tensor.store[width=opt_nelts](
                i+remainder_offset, func[dtype, opt_nelts](simd_data)
            )
        vectorize[remainder_closure, opt_nelts](comps_remainder)
        return result_tensor


    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype], intval: Int) -> Tensor[dtype]:

        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        alias opt_nelts = simdwidthof[dtype]()
        
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data = tensor.load[width=opt_nelts](i)

            result_tensor.store[width=opt_nelts](
                i, func[dtype, opt_nelts](simd_data, intval)
            )

        vectorize[closure, opt_nelts](tensor.num_elements())
        return result_tensor

struct Naive(Backend):
    """
    Naive Backend Struct.

    Just loops for SIMD[Dtype, 1] equations
    """
    
    fn __init__(inout self:Self):
       pass


    
    fn _math_func_1_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            type, simd_w
        ],
    ](self: Self, tensor: Tensor[dtype]) -> Tensor[dtype]:
        """
        Apply a SIMD function of one variable and one return to a tensor

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        
        for i in range(
            tensor.num_elements()
        ):
            var simd_data = func[dtype, 1](tensor.load[width=1](i))
            result_tensor.store[width=1](i, simd_data)
        return result_tensor


    fn _math_func_2_tensor_in_one_tensor_out[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[type, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
        """
        Apply a SIMD function of two variable and one return to a tensor

        Constraints:
            Both tensors must have the same shape

        Parameters:
            dtype: The element type.
            func: the SIMD function to to apply.

        Args:
            tensor1: A tensor
            tensor2: A tensor

        Returns:
            A a new tensor that is tensor with the function func applied.
        """

        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor1.shape())
        
        for i in range(tensor1.num_elements()):
            var simd_data1 = tensor1.load[width=1](i)
            var simd_data2 = tensor2.load[width=1](i)
            result_tensor.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
        return result_tensor


    fn _math_func_compare_2_tensors[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](self:Self,tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[DType.bool]:
        if tensor1.shape() != tensor2.shape():
            with assert_raises():
                raise "Shape Mismatch error shapes must match for this function"
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor1.shape())
        
        for i in range(tensor1.num_elements()):
            var simd_data1 = tensor1.load[width=1](i)
            var simd_data2 = tensor2.load[width=1](i)
            result_tensor.store[width=1](
                i, func[dtype, 1](simd_data1, simd_data2)
            )
        return result_tensor


    fn _math_func_is[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w]) -> SIMD[
            DType.bool, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype]) -> Tensor[DType.bool]:
        var result_tensor: Tensor[DType.bool] = Tensor[DType.bool](tensor.shape())
        

        for i in range(tensor.num_elements()):
            var simd_data = func[dtype, 1](tensor.load[width=1](i))
            result_tensor.store[width=1](i, simd_data)
        return result_tensor


    fn _math_func_simd_int[
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (SIMD[type, simd_w], Int) -> SIMD[
            type, simd_w
        ],
    ](self:Self,tensor: Tensor[dtype], intval: Int) -> Tensor[dtype]:
        var result_tensor: Tensor[dtype] = Tensor[dtype](tensor.shape())
        
        for i in range(tensor.num_elements()):
            var simd_data1 = tensor.load[width=1](i)
            result_tensor.store[width=1](i, func[dtype, 1](simd_data1, intval))
        return result_tensor