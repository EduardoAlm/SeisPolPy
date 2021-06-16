from SeisPolPy import Flinn, Pinnegar, Rstfr, Vidale
import mat4py
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data = mat4py.loadmat('SeisPolPy/tests/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])


def test_rstfr_stft():
    f1 = open("SeisPolPy/tests/outputb64files/outputRstfrB6STFTMajor.txt", "r")
    f2 = open("SeisPolPy/tests/outputb64files/outputRstfrB6STFTMinor.txt", "r")
    out1, out2 = Rstfr.rstfr(sig, "stft") 
    assert out1 == f1.read()
    assert out2 == f2.read()
    f1.close()
    f2.close()

def test_rstfr_stft_condition():
    Rstfr.rstfr(sig, "that's bananas") 


   

"""
def rstfr_s_stft():
    f1 = open("SeisPolPy/tests/outputb64files/outputRstfrB6_S_STFTMajor.txt", "r")
    f2 = open("SeisPolPy/tests/outputb64files/outputRstfrB6_S_STFTMinor.txt", "r")
    out1, out2 = Rstfr.rstfr(sig) 
    assert out1 == f1.read()
    assert out2 == f2.read()
    f1.close()
    f2.close()
"""
