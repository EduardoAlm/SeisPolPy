import SeisPolPy
import mat4py
import numpy as np

data = mat4py.loadmat('examples/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
SeisPolPy.Rstfr.rstfr(sig)