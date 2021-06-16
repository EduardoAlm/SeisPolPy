from SeisPolPy import Flinn, Pinnegar, Rstfr, Vidale
import mat4py
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data = mat4py.loadmat('SeisPolPy/tests/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])

def test_flinn():
    f = open("SeisPolPy/tests/outputb64files/outputFlinnB64.txt", "r")

    assert Flinn.flinn(sig, 50) == f.read()
    f.close()

