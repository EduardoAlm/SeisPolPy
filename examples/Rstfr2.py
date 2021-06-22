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
        #V = np.transpose(V)
        eig_values = np.real(eig_values)
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

def amplitude(zeta, eta, eig_values):
    """
    Creates the amplitude filter.

    :parameter zeta: default value is 0.26. Amplitude filter adjusting parameter.
    :type zeta: float
    :parameter eta: default value is 0.23. Amplitude filter adjusting parameter.
    :type eta: float
    :parameter eig_values: array containing all the eigenvalues for the given signal.
    :type eig_values: array
    :return: array containing the obtained directivity values. 
    """
    
    amp = eig_values[2]
    if amp >= 0 and amp < zeta:
        ampli = 0.
    elif amp > zeta and amp < eta:
        ampli = math.cos(math.pi*(amp-zeta)/2*(eta-zeta))
    elif amp > eta and amp <= 1:
        ampli = 1.
    else:
        raise Exception("Amplitude out of bounds when aplying the rectilinearity filter.")
    return ampli

def directivity_love(gamma, lamb_da, eig_vec3):
    """
    Creates the directivity filter for love waves.

    :parameter gamma: default value is 0.25. Directivity filter adjusting parameter.
    :type alpha: float
    :parameter beta: default value is 0.3. Directivity filter adjusting parameter.
    :type beta: float
    :parameter eig_vec3: array containing all the biggest eigenvectors for the given signal.
    :type eig_vec3: array
    :return: the obtained directivity value for the love wave. 
    """
    print("dir_love")
    b_vec=np.array([[1,0,0],[0,1,0],[0,0,1]], np.int)

    degree_res = np.max(np.abs(np.dot(eig_vec3[2], b_vec[0]))) 
    degree_dir = float("{:.2f}".format(round(degree_res, 2)))      
    if degree_dir >= 0 and degree_dir < gamma:
        degree_res_dir = 1.
    elif degree_dir > gamma and degree_dir < lamb_da:
        degree_res_dir = math.cos(math.pi*(degree_dir-gamma)/2*(lamb_da-gamma))
    elif degree_dir > lamb_da and degree_dir <= 1:
        degree_res_dir = 0.
    else:
        raise Exception("Degree directivity out of bounds when aplying the directionality filter.")

    return degree_res_dir

def directivity_rayleigh(gamma, lamb_da, eig_vec3):
    """
    Creates the directivity filter for raileigh waves.

    :parameter gamma: default value is 0.25. Directivity filter adjusting parameter.
    :type alpha: float
    :parameter beta: default value is 0.3. Directivity filter adjusting parameter.
    :type beta: float
    :parameter eig_vec3: array containing all the biggest eigenvectors for the given signal.
    :type eig_vec3: array
    :return: array containing the obtained directivity values for the rayleigh wave. 
    """
    print("dir_ray")
    b_vec = np.array([[1,0,0],[0,1,0],[0,0,1]], np.int)

    radial = np.max(np.abs(np.dot(eig_vec3, b_vec[1])))
    z = np.max(np.abs(np.dot(eig_vec3, b_vec[2])))
    degree_res = math.sqrt(radial+z)         
    degree_dir = float("{:.2f}".format(round(degree_res, 2)))
    if degree_dir >= 0 and degree_dir < gamma:
        degree_res_dir = 1.
    elif degree_dir > gamma and degree_dir < lamb_da:
        degree_res_dir = math.cos(math.pi*(degree_dir-gamma)/2*(lamb_da-gamma))
    elif degree_dir > lamb_da and degree_dir <= 1:
        degree_res_dir = 0.
    else:
        raise Exception("Degree directivity out of bounds when aplying the directionality filter.")
    return degree_res_dir

def rectilinearity(alpha, beta, eig_values):
    """
    Creates the rectilinearity filter.

    :parameter alpha: default value is 0.1. Rectilinearity filter adjusting parameter.
    :type alpha: float
    :parameter beta: default value is 0.12. Rectilinearity filter adjusting parameter.
    :type beta: float
    :parameter eig_values: array containing all the eigenvalues for the given signal.
    :type eig_values: array
    :return: array containing the obtained rectilinearity values. 
    """
    print("dir_rec")
    degree_res = 1-((eig_values[1]+eig_values[0])/eig_values[2])
    degree_rec = float("{:.2f}".format(round(degree_res, 2)))
    if degree_rec >= -1 and degree_rec < alpha:
        rec = 1.
    elif degree_rec > alpha and degree_rec < beta:
        rec = math.cos(math.pi*(degree_rec-alpha)/2*(beta-alpha))
    elif degree_rec > beta and degree_rec <= 1:
        rec = 0.
    else:
        raise Exception("Degree rectilinearity out of bounds when aplying the rectilinearity filter.")

    return rec

def rstfr(data, alg="stft",filt="love", s=100, n_it=400, alpha=0.1, beta=0.12, gamma=0.25, lamb_da=0.3, zeta=0.26, eta=0.23):
    """
    Obtains semi major, semi minor by implementing an adaptation of pinnegar method which \
    takes advantage of sparsity this method allows for the choice between the normal STFT (Pinnegar Method) \
    and the use of STFT with Sparsity Matrices. \
    Signal in Z, R, T orientation.

    :parameter data: Three component signal data.
    :type data: array  
    :parameter alg: default value is "stft", corresponds to choosing the method STFT. The other option \
        is to give as input "s_stft", which indicates that the chosen method is the sparse STFT.
    :type alg: 
    :parameter filt: default values is "love", corresponds to choosing the type of waves to be filtered, Love \
        or Rayleigh waves, with the available options of "love" and "rayleigh".
    :type filt: string
    :parameter s: default value is 100.
    :type s: int
    :parameter n_it: default value is 400, corresponds to the number of iterations for the softthreshholding. \
        This variable is not used if the chosen method is the normal STFT.
    :parameter alpha: default value is 0.1. Rectilinearity filter adjusting parameter.
    :type alpha: int
    :parameter beta: default value is 0.12. Rectilinearity filter adjusting parameter.
    :type beta: int
    :parameter gamma: default value is 0.25. Directivity filter adjusting parameter.
    :type gamma: int
    :parameter lamb_da: default value is 0.3. Directivity filter adjusting parameter.
    :type lamb_da: int
    :parameter zeta: default value is 0.26. Amplitude filter adjusting parameter.
    :type zeta: int
    :parameter eta: default value is 0.23. Amplitude filter adjusting parameter.
    :type eta: int
    :parameter n_it: default value is 400, corresponds to the number of iterations for the softthreshholding. \
        This variable is not used if the chosen method is the normal STFT.
    :type n_it: int 
    :return: numpy array with semi major, numpy array with semi minor, numpy array with the filtered data and two base64 encoded strings of \
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
    print(semi[0][0,0])
    n2 = len(semi[0, 0]) #last part
    n = len(semi[0]) #middle part
    
    majornorm, minornorm = np.zeros((nf, N)), np.zeros((nf, N))
    
    for i in range(nf):
        for j in range(N):
            # Normalizes the semi major and minor values
            majornorm[i, j], minornorm[i, j] = np.sqrt(np.dot(majo[:, i, j],majo[:, i, j])), np.sqrt(np.dot(mino[:, i, j],mino[:, i, j]))
        print("Normalising the output values ... ",np.round(i*100/nf, 2), "%")

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

    """
    x = np.arange(semi.shape[0])[:, None, None]
    y = np.arange(semi.shape[1])[None, :, None]
    z = np.arange(semi.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)
    c = np.tile(semi.ravel()[:, None], [1, 3])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=semi.ravel())
    plt.show()
    plt.close()
    """
    rec_filter = np.zeros((n2, n2), dtype=np.float16)
    amp_filter = np.zeros((n2, n2), dtype=np.float16)
    dir_love_filter = np.zeros((n2,n2), dtype=np.float16)
    dir_rayleigh_filter = np.zeros((n2, n2), dtype=np.float16)
    for i in range(n):
        for j in range(n2):
            eig_values = np.array([semi[9][0,0],semi[10][0,0],semi[11][0,0]])
            biggesteig_vec=np.array([semi[6][0,0],semi[7][0,0], semi[8][0,0]])        
            rec_filter[j, j] = rectilinearity(alpha, beta, eig_values)
            amp_filter[j, j] = amplitude(zeta, eta, eig_values)

            if filt=="love":
                dir_love_filter[j, j] = directivity_love(gamma, lamb_da, biggesteig_vec)
            elif filt=="rayleigh":
                dir_rayleigh_filter[j, j] = directivity_rayleigh(gamma, lamb_da, biggesteig_vec)
            else:
                raise Exception("The chosen options is not available please check the documentation.")
    if filt=="love":
        rej_love_filter = 1 - np.multiply((1 - rec_filter), (1 - dir_love_filter), (1 - amp_filter))
        ext_love_filter = np.multiply((1 - rec_filter), (1 - dir_love_filter), (1 - amp_filter))
        rejected_x = np.dot(tfrx, rej_love_filter)
        rejected_y = np.dot(tfry, rej_love_filter)
        rejected_z = np.dot(tfrz, rej_love_filter)
        extracted_x = np.dot(tfrx, ext_love_filter)
        extracted_y = np.dot(tfry, ext_love_filter)
        extracted_z = np.dot(tfrz, ext_love_filter)
    elif filt=="rayleigh":
        rej_rayleigh_filter = 1 - np.multiply((1 - rec_filter), (1 - dir_rayleigh_filter), (1 - amp_filter))
        ext_rayleigh_filter = np.multiply((1 - rec_filter), (1 - dir_rayleigh_filter), (1 - amp_filter))
        rejected_x = np.multiply(tfrx, rej_rayleigh_filter)
        rejected_y = np.multiply(tfry, rej_rayleigh_filter)
        rejected_z = np.multiply(tfrz, rej_rayleigh_filter)
        extracted_x = np.multiply(tfrx, ext_rayleigh_filter)
        extracted_y = np.multiply(tfry, ext_rayleigh_filter)
        extracted_x = np.multiply(tfrz, ext_rayleigh_filter)
    else:
        raise Exception("The chosen options is not available please check the documentation.")            
    
    
    plt.figure()
   # f, ax = plt.subplots(3,1) 
   # ax[0].imshow(tfrx)
   # ax[1].imshow(tfry)
   # ax[2].imshow(tfrz)
   # plt.title("stft_s_ist components")
    f, ax2 = plt.subplots(3,1) 
    ax2[0].imshow(rejected_x)
    ax2[1].imshow(rejected_y)
    ax2[2].imshow(rejected_z)
    plt.title("rejection components")
    plt.tight_layout()
    StringIObytes3 = io.BytesIO()
    plt.savefig(StringIObytes3, format='jpg')
  
    StringIObytes3.seek(0)
    b64jpgdataextraction = base64.b64encode(StringIObytes3.read()).decode()
    plt.show()
    plt.close()
    f, ax3 = plt.subplots(3,1) 
    ax3[0].imshow(extracted_x)
    ax3[1].imshow(extracted_y)
    ax3[2].imshow(extracted_z)
    plt.title("extraction components")
    plt.tight_layout()
    StringIObytes4 = io.BytesIO()
    plt.savefig(StringIObytes4, format='jpg')
  
    StringIObytes4.seek(0)
    b64jpgdatarejection = base64.b64encode(StringIObytes4.read()).decode()
    plt.show()
    plt.close()


    print("Execution time:", time.time()-start_time)

    return b64jpgdataMajor, b64jpgdataMinor, b64jpgdataextraction, b64jpgdatarejection
