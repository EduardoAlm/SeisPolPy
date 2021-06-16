from SeisPolPy import Flinn, Pinnegar, Rstfr, Vidale
import mat4py
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data = mat4py.loadmat('SeisPolPy/tests/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])

def test_Pinnegar():
    f1 = open("SeisPolPy/tests/outputb64files/outputPinnegarB64Major.txt", "r")
    f2 = open("SeisPolPy/tests/outputb64files/outputPinnegarB64Minor.txt", "r")

    out1, out2 = Pinnegar.pinnegar(sig) 
    assert out1 == f1.read()
    assert out2 == f2.read()
    f1.close()
    f2.close()