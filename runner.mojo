import base as numojo
from tensor import Tensor
def main():
    var tens = Tensor[DType.float32](10,10)
    tens=tens+numojo.pi/2
    print(numojo.sin[DType.float32](tens))
   