"""
RSTFR Method.

:copyright:
    Eduardo Rodrigues de Almeida
:license:
    The MIT License (MIT)
    Copyright (c) 2021 MrEdwards
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import base64
import io
start_time = time.time()
import scipy.sparse as sp
import forw_op
import adjoint
import math
DTYPE = np.float64
import diags


def soft_threshholding(z, T):
    """
    Soft_threshholding function computes the threshholding to the data.

    :parameter z: array obtained in the calling function
    :type z: array
    :parameter T: array obtained in the calling function
    :type T: array
    :return: array with absolute values
    """
    az = np.abs(z)
    res = np.maximum(az-T, 0)/(np.maximum(az-T,0)+T)*z
    return np.abs(res)

def cross(x1, x2):
    """
    Cross function performs cross correlation.

    :parameter x1: signal component array
    :type x1: array
    :parameter x2: signal component array
    :type x2: array
    :return: numpy array
    """
    x = x1 * np.conj(x2)
    length = len(x1)
    half = np.ceil(length/2).astype(np.int32)
    fstpart = x[1:(half)]
    sndpart = np.flipud(x[(half+1):])
    sumofparts = np.add(fstpart, sndpart)
    corr = np.concatenate(([x[0]], sumofparts))/length
    return corr

def semimm(t, r, z):
    """
    Semi major/minor function which calls the cross coorelation function and performs eigen decomposition.

    :parameter t: numpy array regarding the t signal component.
    :type t: array
    :parameter r: numpy array regarding the r signal component.
    :type r: array
    :parameter z: numpy array regarding the z signal component.
    :type z: array
    :return: numpy array semi, major, minor, major_norm, minor_norm
    """
    length = len(t)
    half = math.ceil(length/2)
    semi = np.zeros((12, half))
    major = np.zeros((3, half))
    minor = np.zeros((3, half))
    tt = cross(t, t).astype(np.float64)
    rr = cross(r, r).astype(np.float64)
    zz = cross(z, z).astype(np.float64)
    tr = cross(t, r).astype(np.float64)
    tz = cross(t, z).astype(np.float64)
    rz = cross(r, z).astype(np.float64)
    for i in range(half):
        cov_matrix = np.array([[tt[i], tr[i], tz[i]],[tr[i], rr[i], rz[i]],[tz[i],rz[i], zz[i]]], dtype='float64')
        eig_values, V = np.linalg.eigh(cov_matrix)
        V = np.transpose(V)
        semi[0:3, i] = V[:, 2]
        semi[3:6, i] = V[:, 1]
        semi[6:9, i] = V[:, 0]
        semi[9:, i] = eig_values
        facc = np.sqrt(length)/np.sqrt(2)
        major[:, i] = np.dot(np.dot(facc, np.sqrt(np.abs(eig_values[2]))), V[:, 2])
        minor[:, i] = np.dot(np.dot(facc, np.sqrt(np.abs(eig_values[1]))), V[:, 1])
    major_norm = major/np.max(np.abs(major))
    minor_norm = minor/np.max(np.abs(minor))
    return semi, major, minor, major_norm, minor_norm

def forward(N, s):
    """
    Forward function calls the Cshared library for the diags function to create a diagonal \
    sparse matrix.
    
    :parameter N: length of component array "x"
    :type N: int
    :parameter s: with default value of S equal to 100.
    :type s: int
    :return: scipy sparse dia matrix. 
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

def stft_s_ist(x, y, z, s, n_it, mu):
    """
    Sparse STFT function which calls the forward function, the Cshared library for the adjoint function \
    and forward operator function, and soft threshholding function.

    :parameter x: numpy array regarding the r signal component.
    :type x: array
    :parameter y: numpy array regarding the t signal component.
    :type y: array
    :parameter z: numpy array regarding the z signal component.
    :type z: array
    :parameter s: with default value of S equal to 100.
    :type s: int 
    :parameter n_it: default value is 400, corresponds to the number of iterations for the softthreshholding. \
        This variable is not used if the chosen method is the normal STFT.
    :type n_it: int 
    :parameter mu: variable with the mu value of 1e-3.
    :type mu: int
    :return: three data arrays corresponding to each component of the signal, after applying the \
        sparse STFT. 
    """
    N = len(x)
    d1, d2, d3 = x, y, z
    dh1, dh2, dh3 = d1, d2, d3
    tfx, tfy, tfz = np.zeros([pow(N, 2), 1], dtype=float), np.zeros([pow(N, 2), 1], dtype=float), np.zeros([pow(N, 2), 1], dtype=float)
    alpha = 1e-2
    
    G = forward(N, s)
    
    lamb_da = np.multiply(np.multiply(mu, np.abs(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
    
    for i in range(n_it):
        U1j = tfx + np.multiply(mu, adjoint.adjoin(G, N, (d1 - forw_op.forw_o(G, N, tfx))))
        U2j = tfy + np.multiply(mu, adjoint.adjoin(G, N, (d2 - forw_op.forw_o(G, N, tfy))))
        U3j = tfz + np.multiply(mu, adjoint.adjoin(G, N, (d3 - forw_op.forw_o(G, N, tfz))))
        tfx = soft_threshholding(U1j, lamb_da)
        tfy = soft_threshholding(U2j, lamb_da)
        tfz = soft_threshholding(U3j, lamb_da)
        d1 = d1 + (dh1 - np.real(forw_op.forw_o(G, N, tfx)))
        d2 = d2 + (dh2 - np.real(forw_op.forw_o(G, N, tfy)))
        d3 = d3 + (dh3 - np.real(forw_op.forw_o(G, N, tfz)))
        #print("Forward/Backward splitting and threshholding... ", np.round(i*100/N, 2), "%")
    
    return np.reshape(tfx, (N, N)), np.reshape(tfy, (N, N)), np.reshape(tfz, (N, N))

def stft(x, s):
    """
    STFT function which calls the forward function and the Cshared library for the adjoint function.

    :parameter x: numpy array regarding one the signal components.
    :type x: array
    :parameter s: with default value of S equal to 100.
    :type s: int
    :return: component data array after applying the STFT.  
    """
    N = len(x)
    G = forward(N, s)
    tfx = adjoint.adjoin(G, N, x)

    return np.reshape(tfx,(N, N))


def rstfr(data, alg="stft", s=100, n_it=400):
    """
    Obtaining semi major, semi minor by implementing an adaptation of pinnegar method which \
    takes advantage of sparsity this method allows for the choice between the normal STFT (Pinnegar Method) \
    and the use of STFT with Sparsity Matrices. \
    Signal in Z, R, T orientation.

    :parameter data: Three component signal data.
    :type data: array  
    :parameter alg: default value is "stft", corresponds to choosing the method STFT. The other option \
        is to give as input "s_stft", which indicates that the chosen method is the sparse STFT.
    :parameter s: default value is 100.
    :type s: int
    :parameter n_it: default value is 400, corresponds to the number of iterations for the softthreshholding. \
        This variable is not used if the chosen method is the normal STFT.
    :type n_it: int 
    :return: numpy array with semi major, numpy array with semi minor and two base64 encoded strings of \
        bytes containing the previous arrays plots.
    """

    sig = data
    start_time = time.time()
 
    plt.rcParams['figure.figsize'] = [13, 8]
    plt.rcParams.update({'font.size': 18})

    t, r, z = sig[0][0:(len(sig[0]))], sig[1][0:(len(sig[0]))], sig[2][0:(len(sig[0]))]

    N, mu = len(r), 1e-3

    if alg == "stft":
        # Obtaining the Short Time Fourier Transform for each component
        tfrx = stft(r, s)  # R component vector 
        tfry = stft(t, s)  # T component vector 
        tfrz = stft(z, s)  # Z component vector 
    elif alg == "s_stft":
        # Obtaining the sparse Short Time Fourier Transform for each component
        tfrx, tfry, tfrz = stft_s_ist(r, t, z, s, n_it, mu)  
    else:
        raise Exception("Please choose an available method (stft/s_stft).")

    nf = np.ceil(N/2).astype(int)
    semi, majo, mino, majon, minon = np.zeros((12,nf,N)), np.zeros((3,nf,N)), np.zeros((3,nf,N)), \
        np.zeros((3,nf,N)), np.zeros((3,nf,N))

    for i in range(N):
        # Generates the semi major/minor
        semi[:,:,i], majo[:,:,i], mino[:,:,i], majon[:,:,i], minon[:,:,i] = semimm(np.multiply(10, tfry[:,i]),\
            np.multiply(10, tfrx[:,i]),np.multiply(10, tfrz[:,i]))
        #print("Generating the semi major/minor ... ",np.round(i*100/N, 2), "%")

    majornorm, minornorm = np.zeros((nf, N)), np.zeros((nf, N))

    for i in range(nf):
        for j in range(N):
            # Normalizes the semi major and minor values
            majornorm[i, j], minornorm[i, j] = np.sqrt(np.dot(majo[:, i, j],majo[:, i, j])), np.sqrt(np.dot(mino[:, i, j],mino[:, i, j]))
        #print("Normalising the output values ... ",np.round(i*100/nf, 2), "%")

    cax = np.max((np.max(majornorm), np.max(minornorm)))
    
    plt.imshow(np.abs(majornorm), cmap='hot', alpha=1, vmin=0, vmax=0.7*cax)
    plt.title('RS-TFR (SM)')    
   
    StringIObytes1 = io.BytesIO()
    plt.savefig(StringIObytes1, format='jpg')

    StringIObytes1.seek(0)
    b64jpgdataMajor = base64.b64encode(StringIObytes1.read()).decode()
    plt.close()
    plt.imshow(np.abs(minornorm), cmap="hot", alpha=1, vmin=0, vmax=0.7*cax)
    plt.title('RS-TFR (Sm)')
   
    StringIObytes2 = io.BytesIO()
    plt.savefig(StringIObytes2, format='jpg')

    StringIObytes2.seek(0)
    b64jpgdataMinor = base64.b64encode(StringIObytes2.read()).decode()
    plt.close()
    print("Execution time:", time.time()-start_time)

    return b64jpgdataMajor, b64jpgdataMinor # np.abs(majornorm), np.abs(minornorm), 
