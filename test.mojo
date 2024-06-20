import numojo

def main():
    var x = numojo.arange[numojo.f32](0,60)
    x.reshape(4,5,3)
    print(x)
    print()
    print(numojo.stats.sum(x,1))
    print()
    print(numojo.stats.mean(x,1))
    