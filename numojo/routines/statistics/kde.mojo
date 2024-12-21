"""
Kernal Density Estimation
"""
from math import exp

from ...core.ndarray import NDArray
from ..math.exp_log import exp as aexp


trait DensityKernal:
    fn kernal[dtype:DType](self:Self,x:Scalar[dtype],h:Scalar[dtype])->Scalar[dtype]:
        pass

    fn kernal[dtype:DType](self:Self,x:NDArray[dtype],h:Scalar[dtype])raises->NDArray[dtype]:
        pass

struct Guassian(DensityKernal):
    fn kernal[dtype:DType](self:Self,x:Scalar[dtype],h:Scalar[dtype])->Scalar[dtype]:
        return exp(
            -((x**2)/
            (2*(h**2))
            ))

    fn kernal[dtype:DType](self:Self,x:NDArray[dtype],h:Scalar[dtype])raises->NDArray[dtype]:
        return aexp(
            -((x**2)/
            (2*(h**2))
            )) 