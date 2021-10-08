import forw_op  
import adjoint 
import numpy as np
cimport numpy as np
from SeisPolPy import Rstfr

cpdef np.ndarray[float, ndim=2] stft_s_isty(np.ndarray[float, ndim=1] y, int s, int n_it, float mu):

    N = len(y)
    tfy = np.zeros([pow(N, 2), 1])
   
    cdef np.ndarray[float, ndim=1] d1 = y
    cdef np.ndarray[float, ndim=1] dh1 = d1
    G = Rstfr.forward(N, s)
    
    lamb_da = np.multiply(np.multiply(mu, np.abs(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
    i=0
    while i < n_it:
        U1j = tfy + np.multiply(mu, adjoint.adjoin(G, N, (d1 - forw_op.forw_o(G, N, tfy))))
       
        tfy = Rstfr.soft_threshholding(U1j, lamb_da)
       
        d1 = d1 + (dh1 - np.real(forw_op.forw_o(G, N, tfy)))
      
        print("iteration ", i, "total ", n_it)
        i+=1
    
    return np.reshape(tfy, (N, N))