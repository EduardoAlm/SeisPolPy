import SeisPolPy
import mat4py
import numpy as np


data = mat4py.loadmat('ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
SeisPolPy.Flinn.flinn(sig, 50)