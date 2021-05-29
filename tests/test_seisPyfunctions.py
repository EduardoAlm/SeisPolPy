#from SeisPolPy import seisPolPyfunctions
from SeisPolPy.Flinn import FlinnMethod
import mat4py
import numpy as np


#def test_flinn():
data = mat4py.loadmat('ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
res = FlinnMethod.flinn(sig)
