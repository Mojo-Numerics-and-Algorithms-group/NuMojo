from .array2d import Array, simdwidthof
from math import mul,sub,add,div,clamp,abs,floor,ceil,ceildiv,trunc,sqrt,rsqrt,exp2,ldexp,exp,frexp,log,log2,copysign,erf,tanh,isclose,all_true,any_true,none_true,reduce_bit_count,iota,is_power_of_2,is_odd,is_even,fma,reciprocal,identity,greater,greater_equal,less,less_equal,equal,not_equal,select,max,min,pow,div_ceil,align_down,align_up,acos,asin,atan,atan2,cos,sin,tan,acosh,asinh,atanh,cosh,sinh,expm1,log10,log1p,logb,cbrt,hypot,erfc,lgamma,tgamma,nearbyint,rint,round,remainder,nextafter,j0,j1,y0,y1,scalb,gcd,lcm,factorial,nan,isnan

struct linalg[dtype:DType, opt_nelts:Int]:
    fn partial_pivot(self, inout A:Array[dtype, opt_nelts], b:Array[dtype, opt_nelts])raises:
        let tol: SIMD[dtype, 1] =1e-6
        let n: Int = A.cols
        for i in range(n):
            var max_val = abs(A[i, i])
            var max_col = i
            
            # Find the maximum absolute value in the current row after the current column
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    max_col = j
            
            # Check if the maximum value is very close to zero
            if max_val < tol:
                raise Error("Pivot element is too close to zero. Algorithm may fail.")
            
            # Swap columns if necessary
            if max_col != i:
                let temp = A[0:n, i]
                A[0:n, i] = A[0:n, max_col]
                A[0:n, max_col] = temp
                let temp2 = b[i]
                b[i] = b[max_col]
                b[max_col] = temp2
                
            for j in range(i + 1, n):
                let factor = A[j, i] / A[i, i]
                for k in range(i, n):
                    A[j, k] -= factor * A[i, k]
                b[j] -= factor * b[i]
                
        # return A, b

    fn lu_decomposition(self,A:Array[dtype, opt_nelts])raises->(Array[dtype, opt_nelts],Array[dtype, opt_nelts]):
        let n: Int = A.cols
        let L :Array[dtype, opt_nelts] = Array[dtype, opt_nelts](A.rows,A.cols)
        var z: Array[dtype, opt_nelts] = Array[dtype, opt_nelts](n, 1)
        for i in range(A.cols):
            for j in range(A.rows):
                if i == j:
                    L[i,j]=1
        var U:Array[dtype, opt_nelts] = A
        let P:Array[dtype, opt_nelts] = L
        
        for i in range(n):
            self.partial_pivot(U, z)  # Apply partial pivoting to U
            let pivot = U[i, i]
            
            for j in range(i + 1, n):
                let factor = U[j, i] / pivot
                L[j, i] = factor
                for k in range(i, n):
                    U[j, k] -= factor * U[i, k]
                
        return L, U
    fn gaussian_elimination_det(self, A:Array[dtype, opt_nelts])raises->SIMD[dtype, 1]:
        let n: Int = A.cols
        var det: SIMD[dtype, 1] = 1.0
        var L = A
        for i in range(n):
            let pivot: SIMD[dtype, 1] = L[i, i]

            if pivot == 0:
                raise Error("A is singular, determinant cannot be calculated.")

            det *= pivot

            # Divide the current row by the pivot
            L[0:n, i] /= pivot

            # Eliminate other rows
            for j in range(i + 1, n):
                let factor: SIMD[dtype, 1] = L[i,j]
                L[0:n,j] -= L[0:n,i] * factor

        return det