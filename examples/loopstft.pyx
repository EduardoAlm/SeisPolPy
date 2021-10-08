import forw_op  
import adjoint 
import numpy as np
cimport numpy as np

cpdef np.ndarray[float, ndim=2] loop(G, np.ndarray[double, ndim=1] x, int N, lamb_da, int n_it, float mu):

    tfx = np.zeros([pow(N, 2), 1])
   
    cdef np.ndarray[double, ndim=1] d1 = x
    cdef np.ndarray[double, ndim=1] dh1 = d1
    
    i=0
    while i < n_it:
        U1j = tfx + np.multiply(mu, adjoint.adjoin(G, N, (d1 - forw_op.forw_o(G, N, tfx))))
        az = np.abs(U1j)
        res = np.maximum(az-lamb_da, 0)/(np.maximum(az-lamb_da,0)+lamb_da)*U1j
        tfx = np.abs(res)
       
        d1 = d1 + (dh1 - np.real(forw_op.forw_o(G, N, tfx)))
      
        print("iteration ", i, "total ", n_it)
        i+=1
    
    return np.reshape(tfx, (N, N))