import base as numojo
from tensor import Tensor
import math
def main():
    var tens1 = Tensor[DType.float32](10,10)
    var tens2 = Tensor[DType.float32](10,10)
    var tens3 = Tensor[DType.index](10,10)
    tens3= tens3 + SIMD[DType.index,1](394)
    tens1=tens1+numojo.pi/2
    tens2=tens2+numojo.pi
    print(numojo.mod(tens1,tens2))
    print(numojo.pow(tens1,5))
    print(numojo.rotate_bits_left[2](tens3))
    # print(numojo.sin[DType.float32](tens))
    # The standard library tensor and simd bool is very broken in terms of setting values
    # var booltens = Tensor[DType.bool](32)
  
    # for i in range(32):
    #     booltens[i] = True
    # print(booltens)
    # booltens.data().store[width=16](16,SIMD[DType.bool,1](True))
    # booltens.data().store[width=16](0,SIMD[DType.bool,1](True))
        # print(booltens[i])
    # booltens[249]=False
    # print(booltens)
    # booltens.data().store[width=1](17,SIMD[DType.bool,1](False))
    # print(booltens.load[width=32](0))
    # print(numojo.none_true(booltens))
   