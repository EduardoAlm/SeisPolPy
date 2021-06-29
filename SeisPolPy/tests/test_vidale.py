from SeisPolPy import Vidale
import mat4py
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def test_vidale():
    data = mat4py.loadmat('SeisPolPy/tests/dataACRG.mat') # seismic data
    sig = np.array([data['t'], data['r'], data['z']])
    f1 = open("SeisPolPy/tests/outputb64files/outputVidaleB641.txt", "r+")

    out1, out2 = Vidale.vidale(sig, 50) 
    
    assert out1 == f1.read()
    
    f2 = open("SeisPolPy/tests/outputb64files/outputVidaleB642.txt", "r+")

    assert out2 == f2.read()
    f1.close()
    f2.close()