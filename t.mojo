import numojo as nm 

def main():
    arr = nm.arange[nm.i64,out_dtype=nm.f64](0,1000)
    arr.reshape(10,10,10)
    # print(nm.greater(arr,arr-1.0))
    # print(arr.min(axis=0))
    print(nm.stats.sin[nm.f64](arr))