import base as numojo
from tensor import Tensor
import math
def main():
    var tens = Tensor[DType.float32](10,10)
    tens=tens+numojo.pi/2
    print(numojo.sin[DType.float32](tens))
    # The standard library tensor and simd bool is very broken in terms of setting values
    # var booltens = Tensor[DType.bool](32)
    # booltens.
    # for i in range(32):
    # booltens.data().store[width=16](16,SIMD[DType.bool,1](True))
    # booltens.data().store[width=16](0,SIMD[DType.bool,1](True))
        # print(booltens[i])
    # booltens[249]=False
    # print(booltens)
    # booltens.data().store[width=1](17,SIMD[DType.bool,1](False))
    # print(booltens.load[width=16](16))
    # print(numojo.none_true(booltens))
   