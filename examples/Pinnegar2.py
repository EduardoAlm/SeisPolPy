"""

SeisPolPy.Pinnegar module
-------------------------

.. automodule:: SeisPolPy.Pinnegar
   :members:
   :undoc-members:
   :show-inheritance:

"""
"""
Pinnegar Method.

:copyright:
    Eduardo Rodrigues de Almeida
:license:
    The MIT License (MIT) 
    Copyright (c) 2021 MrEdwards
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from numpy.core.fromnumeric import size
start_time = time.time()
import scipy.sparse as sp
import adjoint
DTYPE = np.float64
import diags
import io 
import base64

def forward(N, s):
    """
    Forward function calls the Cshared library for the diags function to create a diagonal \
    sparse matrix.
    
    :param N: length of component array "x"
    :type N: int
    :param s: default value of S equal to 100.
    :type s: int
    :returns: scipy sparse dia matrix. 
    """
    id = np.linspace(0, N-1, N, dtype="int32")
    w = np.zeros([N, N], dtype=DTYPE)

    if type(s) == int:
        sj = s * np.ones([N, 1]).astype(int)
    elif len(s) != N:
        raise Exception('length of "s" should be one or the same as length of data ')

    for i in range(N-1):
        w[:, i] = np.exp(np.multiply(-2 * pow(np.pi, 2), pow(id - i, 2)) / pow(sj[i], 2))

    offset = np.linspace(0, pow(N, 2) - N, N, dtype="int32")
    y = np.transpose(w)
    res = diags.diagonal(y.tolist(), offset,  N, pow(N, 2), dtype=DTYPE)

    return sp.dia_matrix((res, offset), shape=(N, pow(N, 2)))

def stft(x, s):
    """
    STFT function which calls the forward function and the Cshared library for the adjoint function.

    :param x: numpy array regarding one the signal components.
    :type x: array
    :param s: default value of S equal to 100.
    :type s: int
    :returns: component data array after applying the STFT.  
    """
    N = len(x)
    G = forward(N, s)
    tfx = adjoint.adjoin(G, N, x)

    return np.reshape(tfx,(N, N))

def pinnegar(data, s=100):
    """
    Obtaining semi major, semi minor, inclination (dip), pitch (omega), phase (phi) and strike (ohm) by implementing \
    the method designed by C. R. Pinnegar. \
    Signal in Z, R, T orientation.

    :parameter data: three component signal data.
    :type data: array   
    :parameter s: default value is 100.
    :type s: int
    :return: numpy array with semi major, numpy array with semi minor, numpy array with dip, \
    numpy array with pitch, numpy array with phase and a base64 encoded string of bytes containing the previous \
    arrays plots.
    """
    sig = data

    plt.rcParams['figure.figsize'] = [16, 12]
    plt.rcParams.update({'font.size': 13})

    length = len(sig[0])
   
    # Obtaining the Short Time Fourier Transform for each component
    tfrx = stft(sig[1][0:(length)], s)  # R component vector 
    tfry = stft(sig[0][0:(length)], s)  # T component vector 
    tfrz = stft(sig[2][0:(length)], s)  # Z component vector 

    sminor = np.zeros((len(tfrx), length))
    smajor = np.zeros((len(tfrx), length))
    inclin = np.zeros((len(tfrx), length))  
    omega = np.zeros((len(tfrx), length))  
    phi = np.zeros((len(tfrx), length))  
    ohm = np.zeros((len(tfrx), length))  
    omega0 = np.zeros((len(tfrx), length))
    phi0 = np.zeros((len(tfrx), length))

    i=0
    while i < (length):
        X = tfrx[:,i]
        Y = tfry[:,i]
        Z = tfrz[:,i]

        # Obtaining the real part of each component
        XR = np.real(X)
        YR = np.real(Y)
        ZR = np.real(Z)

        # Obtaining the imaginary part of each component
        XI = np.imag(X)
        YI = np.imag(Y)
        ZI = np.imag(Z)

        A = pow(XR, 2) + pow(XI, 2) + pow(YR, 2) + pow(YI, 2) \
            + pow(ZR, 2) + pow(ZI, 2)
        B = pow(XR, 2) - pow(XI, 2) + pow(YR, 2) - pow(YI, 2) \
            + pow(ZR, 2) - pow(ZI, 2)
        C = np.multiply(-2, ((XR * XI) + (YR * YI) + (ZR * ZI)))
        D = np.sqrt(pow(B, 2) + pow(C, 2))
        a = np.multiply((1 / np.sqrt(2)), (np.sqrt(A + D)))
        b = np.multiply((1 / np.sqrt(2)), (np.sqrt(np.abs(A - D))))
        I = np.arctan2(np.sqrt(pow((ZR * YI - ZI * YR), 2) + \
                    pow((ZR * XI - ZI * XR), 2)), (YR * XI - YI * XR))
        OHM = np.arctan2((ZR * YI - ZI * YR), (ZR * XI - ZI * XR))
        ph0 = np.multiply(0.5, np.arctan2(C, B))
        K = b * (ZR * np.cos(ph0) - ZI * np.sin(ph0))
        K = np.real(K)
        L = -a * (ZR * np.sin(ph0) + ZI * np.cos(ph0))
        L = np.real(L)
        w0 = np.arctan2(K, L)
        w = w0 - np.pi * ((np.sign(w0) - 1) / 2)
        ph = ph0 + (np.pi * ((np.sign(w0) - 1) / 2) * np.sign(ph0))

        smajor[:, i] = a  # semi major
        sminor[:, i] = b  # semi minor
        inclin[:, i] = I  # inclination
        omega[:, i] = w  # pitch (the angle between the ascending node and the position of maximum displacement)
        phi[:, i] = ph  # phase
        ohm[:, i] = OHM  # strike (the azimuth of the ascending node)
        omega0[:, i] = w0
        phi0[:, i] = ph0
        i+=1

    half = math.floor(len(smajor)/2)
    
    ab, bc= np.shape(np.abs(np.transpose(smajor[:][:,0:half])))
    fdim = np.linspace(0, ab, 6)
    fig,(ax1) = plt.subplots(1,1)
    
    a = ax1.imshow(np.abs(np.transpose(smajor[:][:,0:half])), aspect='auto')
    ax1.set_yticks(fdim)
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Semi Major")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')


    StringIObytes = io.BytesIO()
    plt.savefig(StringIObytes, format='jpg')
    plt.show()
    plt.close()
    StringIObytes.seek(0)
    b64jpgdataM = base64.b64encode(StringIObytes.read()).decode()


    fig,(ax2) = plt.subplots(1,1)
    b = ax2.imshow(np.abs(np.transpose(sminor[:][:,0:half])),aspect='auto')
    ax2.set_yticks(fdim)
    ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar2 = fig.colorbar(b)
    cbar2.set_label("Amplitude (μm)")
    plt.title("Semi Minor")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes2 = io.BytesIO()
    plt.savefig(StringIObytes2, format='jpg')
    plt.show()
    plt.close()
    StringIObytes2.seek(0)
    b64jpgdatam = base64.b64encode(StringIObytes2.read()).decode()

    print("Time of execution", time.time()-start_time)

    return b64jpgdataM, b64jpgdatam #smajor, sminor, inclin, omega, phi, ohm, 
