import SeisPolPy
import mat4py
import numpy as np


data = mat4py.loadmat('tests/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
g1, g2_z, g2_r, g2_t, b64jpgdata = SeisPolPy.FlinnMethod.flinn(sig)