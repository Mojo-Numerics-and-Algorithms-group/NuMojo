import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
"""
TODO
Tell Modular something is wrong with simd bools store method
"""
alias sbool = SIMD[DType.bool,1]
fn all_true(tensor:Tensor[DType.bool])->Bool:
    """
    Returns true if all elements of a bool tensor are True.
    """
    var result: sbool = True
    alias opt_nelts = simdwidthof[DType.bool]()
    for i in range(0, opt_nelts*(tensor.num_elements()//opt_nelts), opt_nelts):
        var simd_data = tensor.load[width=opt_nelts](i)
        print(simd_data)
        var section_all_true = math.all_true(simd_data)
        result = math.logical_and(sbool(result),sbool(section_all_true))
    if tensor.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor.num_elements()//opt_nelts), tensor.num_elements()):
            var simd_data = tensor.load[width=1](i)
            result = math.logical_and(sbool(result),simd_data)
    return result

def any_true(tensor:Tensor[DType.bool]):
    """
    Returns true if any elements of a bool tensor are True.
    """
    var result: sbool = False
    alias opt_nelts = simdwidthof[DType.bool]()
    for i in range(0, opt_nelts*(tensor.num_elements()//opt_nelts), opt_nelts):
        var simd_data = tensor.load[width=opt_nelts](i)
        var section_any_true = math.any_true(simd_data)
        result = result or section_any_true
    if tensor.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor.num_elements()//opt_nelts), tensor.num_elements()):
            var simd_data = tensor.load[width=1](i)
            result = result or simd_data
    return result

def none_true(tensor:Tensor[DType.bool])->Bool:
    """
    None true is equivelent to not any_true and means all false.
    """
    return not any_true(tensor)
# def main():
#     tens = Tensor[DType.bool](100)



