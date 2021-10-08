"""
RSTFR Method.

:copyright:
    Eduardo Rodrigues de Almeida
:license:
    The MIT License (MIT)
    Copyright (c) 2021 MrEdwards
"""
from numpy import shape, savetxt, loadtxt, int16, maximum, conj, ceil, flipud, add, concatenate, reshape, int32, zeros, array, real, sqrt, dot, multiply, linspace, exp, float64, ones, pi, transpose, float16

from numpy.linalg import eigh
from numpy import abs as abso
from numpy import max as maxim
from numpy import round as roundo
import matplotlib.pyplot as plt
import time
import base64

import io
start_time = time.time()
from scipy.sparse import dia_matrix
from scipy.fftpack import ifft
import forw_op  
import adjoint 
import I_stft
import loopstft
from math import cos

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
    az = abso(z)
    res = maximum(az-T, 0)/(maximum(az-T,0)+T)*z
    return abso(res)

def cross(x1, x2):
    """
    Cross function performs cross correlation.

    :parameter x1: signal component array
    :type x1: array
    :parameter x2: signal component array
    :type x2: array
    :return: numpy array
    """
    x = x1 * conj(x2)
    length = len(x1)
    half = ceil(length/2).astype(int32)
    fstpart = x[1:(half)]
    sndpart = flipud(x[(half+1):])
    sumofparts = add(fstpart, sndpart)
    corr = concatenate(([x[0]], sumofparts))/length
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
    half = ceil(length/2).astype(int16)
    print(half)
    semi,major,minor = zeros((12, half)),zeros((3, half)),zeros((3, half))
    tt, rr,zz,tr,tz,rz, i= cross(t, t),cross(r, r),cross(z, z),cross(t, r),cross(t, z),cross(r, z),0
 
    while i < half:
        eig_values, V = eigh(array([[tt[i], tr[i], tz[i]],[tr[i], rr[i], rz[i]],[tz[i],rz[i], zz[i]]], dtype='float64'))
        #V = np.transpose(V)
        eig_values = real(eig_values)
        semi[0:3, i] = V[:, 2]
        semi[3:6, i] = V[:, 1]
        semi[6:9, i] = V[:, 0]
        semi[9:, i] = eig_values
        facc = sqrt(length)/sqrt(2)
        major[:, i], minor[:, i] = dot(dot(facc, sqrt(abso(eig_values[2]))), V[:, 2]), dot(dot(facc, sqrt(abso(eig_values[1]))), V[:, 1])
         
        i+=1
        
    return semi, major, minor, major/maxim(abso(major)), minor/maxim(abso(minor))

def forward(N, s):
    """import
    :parameter N: length of component array "x"
    :type N: int
    :parameter s: with default value of S equal to 100.
    :type s: int
    :return: scipy sparse dia matrix. 
    """
    ids, w = linspace(0, N-1, N, dtype=int32), zeros([N, N], dtype=float64)
    
    if type(s) == int:
        sj = s * ones([N, 1]).astype(int)
    elif len(s) != N:
        raise Exception('length of "s" should be one or the same as length of data ')
    i=0
    while i < N:
        w[:, i] = exp(multiply(-2 * pow(pi, 2), pow(ids - i, 2)) / pow(sj[i], 2))
        i+=1

    offset = linspace(0, pow(N, 2) - N, N, dtype=int32)

    y = transpose(w)

    res = diags.diagonal(y.tolist(), offset,  N, pow(N, 2), dtype=float64)
    return dia_matrix((res, offset), shape=(N, pow(N, 2)))

def stft_s_istx(x, s, n_it, mu):
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
    d1= x
    G = forward(N, s)
    lamb_da = multiply(multiply(mu, abso(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
    
    return loopstft.loop(G, x, N, lamb_da, n_it, mu)

def stft_s_isty( y, s, n_it, mu):
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
    N = len(y)
    d1= y
    G = forward(N, s)
    lamb_da = multiply(multiply(mu, abso(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
    
    return loopstft.loop(G, y, N, lamb_da, n_it, mu)


def stft_s_istz(z, s, n_it, mu):
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
    N = len(z)
    d1= z
    G = forward(N, s)
    lamb_da = multiply(multiply(mu, abso(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
    
    return loopstft.loop(G, z, N, lamb_da, n_it, mu)


def stft(x, s):
    """
    STFT function which calls the forward function and the Cshared library for the adjoint function.

    :parameter x: numpy array regarding one the signal components.
    :type x: array
    :parameter s: with default value of S equal to 100.
    :type s: int
    :return: component data array after applying the STFT.  
    """
    N = len(x), 
    G = forward(N, s)
    tfx = adjoint.adjoin(G, N, x)

    return reshape(tfx,(N, N))

def amplitude(zeta, eta, eig_values, L):
    """
    Creates the amplitude filter.

    :parameter zeta: default value is 0.26. Amplitude filter adjusting parameter.
    :type zeta: float
    :parameter eta: default value is 0.23. Amplitude filter adjusting parameter.
    :type eta: float
    :parameter eig_values: array containing all the eigenvalues for the given signal.
    :type eig_values: array
    :return:  the obtained amplitude value. 
    """
    
    amp_attr = sqrt(2)*eig_values[2]/L

    degree_amp = float("{:.2f}".format(roundo(amp_attr, 2))) 
    if degree_amp >= 0 and degree_amp < zeta:
        ampli = 0.
    elif degree_amp > zeta and degree_amp < eta:
        ampli = cos(pi*(degree_amp-zeta)/2*(eta-zeta))
    elif degree_amp > eta and degree_amp <= 1:
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
    b_vec=array([[1,0,0],[0,1,0],[0,0,1]], int)

    degree_res = maxim(abso(dot(eig_vec3[2], b_vec[0]))) 
    degree_dir = float("{:.2f}".format(roundo(degree_res, 2)))      
    if degree_dir >= 0 and degree_dir < gamma:
        degree_res_dir = 1.
    elif degree_dir > gamma and degree_dir < lamb_da:
        degree_res_dir = cos(pi*(degree_dir-gamma)/2*(lamb_da-gamma))
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
    b_vec = array([[1,0,0],[0,1,0],[0,0,1]], int)

    radial, z = maxim(abso(dot(eig_vec3, b_vec[1]))), maxim(abso(dot(eig_vec3, b_vec[2])))
    
    degree_res = sqrt(radial+z)         
    degree_dir = float("{:.2f}".format(roundo(degree_res, 2)))
    if degree_dir >= 0 and degree_dir < gamma:
        degree_res_dir = 1.
    elif degree_dir > gamma and degree_dir < lamb_da:
        degree_res_dir = cos(pi*(degree_dir-gamma)/2*(lamb_da-gamma))
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
    :return: the obtained rectilinearity value. 
    """
    degree_res = 1-((eig_values[1]+eig_values[0])/eig_values[2])
    degree_rec = float("{:.2f}".format(roundo(degree_res, 2)))
    if degree_rec >= -1 and degree_rec < alpha:
        rec = 1.
    elif degree_rec > alpha and degree_rec < beta:
        rec = cos(pi*(degree_rec-alpha)/2*(beta-alpha))
    elif degree_rec > beta and degree_rec <= 1:
        rec = 0.
    else:
        raise Exception("Degree rectilinearity out of bounds when aplying the rectilinearity filter.")

    return rec

    
import math
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

    sig, start_time = data, time.time()

    plt.rcParams['figure.figsize'] = [13, 8]
    plt.rcParams.update({'font.size': 18})

    t, r, z = sig[0][0:(len(sig[0]))], sig[1][0:(len(sig[0]))], sig[2][0:(len(sig[0]))]

    fig,(ax3) = plt.subplots()
    ax3.set_title("Extraction component T") 

    a = ax3.plot(t)
    plt.show()
    plt.close()
    fig,(ax3) = plt.subplots()
    ax3.set_title("Extraction component R") 

    a = ax3.plot(r)
    plt.show()
    plt.close()
    fig,(ax3) = plt.subplots()
    ax3.set_title("Extraction component Z") 

    a = ax3.plot(z)
    plt.show()
    plt.close()

    N, mu = len(r), 1e-3
    if N%2==1:
        t, r, z = sig[0][0:(len(sig[0]))-1], sig[1][0:(len(sig[0]))-1], sig[2][0:(len(sig[0]))-1]
        N=len(r)
    if alg == "stft":
        # Obtaining the Short Time Fourier Transform for each component
        tfrx, tfry, tfrz = stft(r, s),  stft(t, s), stft(z, s) # R component vector 
    
    elif alg == "s_stft":
        # Obtaining the sparse Short Time Fourier Transform for each component
        tfrx = stft_s_istx(r, s, n_it, mu)
        tfry = stft_s_isty(t, s, n_it, mu)
        tfrz = stft_s_istz(z, s, n_it, mu)
        savetxt('tfrx.txt', tfrx, delimiter=',')
        savetxt('tfry.txt', tfry, delimiter=',')
        savetxt('tfrz.txt', tfrz, delimiter=',')
        print("s_stft: already loaded")
    else:
        raise Exception("Please choose an available method (stft/s_stft).")

    tfrx = loadtxt('tfrx.txt', dtype='float64', delimiter=',')
    tfry = loadtxt('tfry.txt', dtype='float64', delimiter=',')
    tfrz = loadtxt('tfrz.txt', dtype='float64', delimiter=',')

    nf = ceil(N/2).astype(int)
    semi, majo, mino, majon, minon,i  = zeros((12,nf,N)), zeros((3,nf,N)), zeros((3,nf,N)), \
        zeros((3,nf,N)), zeros((3,nf,N)), 0
    
    while i < N:
        # Generates the semi major/minor
        semi[:,:,i], majo[:,:,i], mino[:,:,i], majon[:,:,i], minon[:,:,i] = semimm(multiply(10, tfry[:,i]),\
            multiply(10, tfrx[:,i]), multiply(10, tfrz[:,i]))
        i+=1
      
        print("Generating the semi major/minor ... ", roundo(i*100/N, 2), "%")

    n2, n = len(semi[0, 0]), len(semi[0])
   
    
    majornorm, minornorm, i, j= zeros((nf, N)), zeros((nf, N)), 0, 0
     
    while i < nf:
        while j < N:
            
            # Normalizes the semi major and minor values
            majornorm[i, j], minornorm[i, j] = sqrt(dot(majo[:, i, j],majo[:, i, j])), sqrt(dot(mino[:, i, j],mino[:, i, j]))
            j += 1
        j=0
        i+=1
        print("Normalising the output values ... ",roundo(i*100/nf, 2), "%")

    cax = maxim((maxim(majornorm), maxim(minornorm)))
    half = math.floor(len(majornorm)/2)
    ab, bc= shape(abso(transpose(majornorm[:][:,0:half])))
    fdim = linspace(0, ab, 6)
   
    fig,(ax1) = plt.subplots(1,1)
    b = ax1.imshow(abso(transpose(majornorm)), aspect='auto',cmap='hot', alpha=1, vmin=0, vmax=0.7*cax)
    ax1.set_yticks(fdim)
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar1 = fig.colorbar(b)
    cbar1.set_label("Amplitude (μm)")
    plt.title('RS-TFR (SM)')    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes1 = io.BytesIO()
    plt.savefig(StringIObytes1, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes1.seek(0)
    b64jpgdataMajor = base64.b64encode(StringIObytes1.read()).decode()
    
    fig,(ax2) = plt.subplots(1,1)
    b = ax2.imshow(abso(transpose(minornorm)), aspect='auto', cmap='hot', alpha=1, vmin=0, vmax=0.7*cax)
    ax2.set_yticks(fdim)
    ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar2 = fig.colorbar(b)
    cbar2.set_label("Amplitude (μm)")
    plt.title('RS-TFR (Sm)')    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes2 = io.BytesIO()
    plt.savefig(StringIObytes2, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes2.seek(0)
    b64jpgdataMinor = base64.b64encode(StringIObytes2.read()).decode()
    
    rec_filter, amp_filter, dir_love_filter, dir_rayleigh_filter, i, j= zeros((n2, n2), dtype=float16), zeros((n2, n2), dtype=float16), zeros((n2,n2), dtype=float16), zeros((n2, n2), dtype=float16), 0, 0
    
    while i < n2:
        while j < n2:
            eig_values = array([semi[9][0,0],semi[10][0,0],semi[11][0,0]])
            biggesteig_vec=array([semi[6][0,0],semi[7][0,0], semi[8][0,0]])        
            rec_filter[i, j] = rectilinearity(alpha, beta, eig_values)
            amp_filter[i, j] = amplitude(zeta, eta, eig_values, N)

            if filt=="love":
                dir_love_filter[i, j] = directivity_love(gamma, lamb_da, biggesteig_vec)
            elif filt=="rayleigh":
                dir_rayleigh_filter[i, j] = directivity_rayleigh(gamma, lamb_da, biggesteig_vec)
            else:
                raise Exception("The chosen options is not available please check the documentation.")
            j += 1
        print("Filtering the output values ... ",roundo(i*100/n2, 2), "%")
        j=0
        i+=1
    if filt=="love":
        rej_love_filter = 1 - multiply((1 - rec_filter), (1 - dir_love_filter), (1 - amp_filter))
        ext_love_filter = (1 - rec_filter)
        rejected_x = multiply(rej_love_filter,tfrx)
        rejected_y = multiply(rej_love_filter,tfry)
        rejected_z = multiply(rej_love_filter,tfrz)
        extracted_x = multiply(ext_love_filter,tfrx)
        extracted_y = multiply(ext_love_filter,tfry)
        extracted_z = multiply(ext_love_filter,tfrz)
    elif filt=="rayleigh":
        rej_rayleigh_filter = 1 - multiply((1 - rec_filter), (1 - dir_rayleigh_filter), (1 - amp_filter))
        ext_rayleigh_filter = (1 - rec_filter)
        rejected_x = multiply(rej_rayleigh_filter,tfrx)
        rejected_y = multiply(rej_rayleigh_filter,tfry)
        rejected_z = multiply(rej_rayleigh_filter,tfrz)
        extracted_x = multiply(ext_rayleigh_filter,tfrx)
        extracted_y = multiply(ext_rayleigh_filter,tfry)
        extracted_z = multiply(ext_rayleigh_filter,tfrz)
    else:
        raise Exception("The chosen options is not available please check the documentation.")            
    
    
    ab, bc= shape(abso(transpose(rejected_x[:][:,0:half])))
    fdim = linspace(0, ab, 6)
    
    fig,(ax1) = plt.subplots(1,1)    
    
    a = ax1.imshow(abso(transpose(rejected_x)), aspect='auto')
    ax1.set_yticks(fdim)
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Rejection component T")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes1 = io.BytesIO()
    plt.savefig(StringIObytes1, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes1.seek(0)
    b64jpgdatarejection1 = base64.b64encode(StringIObytes1.read()).decode()


    fig,(ax2) = plt.subplots(1,1)
    
    a = ax2.imshow(abso(transpose(rejected_y)), aspect='auto')
    ax2.set_yticks(fdim)
    ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Rejection component R")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes2 = io.BytesIO()
    plt.savefig(StringIObytes2, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes2.seek(0)
    b64jpgdatarejection2 = base64.b64encode(StringIObytes2.read()).decode()


    fig,(ax3) = plt.subplots(1,1)

    a = ax3.imshow(abso(transpose(rejected_z)), aspect='auto')
    ax3.set_yticks(fdim)
    ax3.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Rejection component Z")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes3 = io.BytesIO()
    plt.savefig(StringIObytes3, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes3.seek(0)
    b64jpgdatarejection3 = base64.b64encode(StringIObytes3.read()).decode()

    fig,(ax1) = plt.subplots(1,1)
    
    a = ax1.imshow(abso(transpose(extracted_x)), aspect='auto')
    ax1.set_yticks(fdim)
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Extraction component T")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes1 = io.BytesIO()
    plt.savefig(StringIObytes1, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes1.seek(0)
    b64jpgdataextraction1 = base64.b64encode(StringIObytes1.read()).decode()

    fig,(ax2) = plt.subplots(1,1)

    a = ax2.imshow(abso(transpose(extracted_y)), aspect='auto')
    ax2.set_yticks(fdim)
    ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Extraction component R")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes2 = io.BytesIO()
    plt.savefig(StringIObytes2, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes2.seek(0)
    b64jpgdataextraction2 = base64.b64encode(StringIObytes2.read()).decode()

    fig,(ax3) = plt.subplots(1,1)

    a = ax3.imshow(abso(transpose(extracted_z)), aspect='auto')
    ax3.set_yticks(fdim)
    ax3.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar = fig.colorbar(a)
    cbar.set_label("Amplitude (μm)")
    plt.title("Extraction component Z")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    StringIObytes3 = io.BytesIO()
    plt.savefig(StringIObytes3, format='jpg')
    #plt.show()
    plt.close()
    StringIObytes3.seek(0)
    b64jpgdataextraction3 = base64.b64encode(StringIObytes3.read()).decode()

    N = len(rejected_x)
    G = forward(N, s)

    recx = I_stft.i_stft(extracted_x, G, N)
    recy = I_stft.i_stft(extracted_y, G, N)
    recz = I_stft.i_stft(extracted_z, G, N)

    fig,(ax3) = plt.subplots()
    timelin=linspace(0, len(recx),len(recx))
    ax3.set_title("Extraction component X") 

    a = ax3.plot(timelin, recx)
    #ax3.set_yticks(fdim)
    #ax3.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    #cbar = fig.colorbar(a)
    #cbar.set_label("Amplitude (μm)")
    
#    ax3.xlabel('Time (s)')
#    ax3.ylabel('Velocity (m/s)')

    StringIObytes3 = io.BytesIO()
    plt.savefig(StringIObytes3, format='jpg')
    plt.show()
    plt.close()
    StringIObytes3.seek(0)
    b64jpgdataextraction3 = base64.b64encode(StringIObytes3.read()).decode()

    fig,(ax3) = plt.subplots()
    ax3.set_title("Extraction component Y") 

    a = ax3.plot(timelin, recy)

    StringIObytes3 = io.BytesIO()
    plt.savefig(StringIObytes3, format='jpg')
    plt.show()
    plt.close()
    StringIObytes3.seek(0)
    b64jpgdataextraction3 = base64.b64encode(StringIObytes3.read()).decode()
    
    fig,(ax3) = plt.subplots()
    ax3.set_title("Extraction component z") 

    a = ax3.plot(timelin, recz)

    StringIObytes3 = io.BytesIO()
    plt.savefig(StringIObytes3, format='jpg')
    plt.show()
    plt.close()
    StringIObytes3.seek(0)
    b64jpgdataextraction3 = base64.b64encode(StringIObytes3.read()).decode()

    print("Execution time:", time.time()-start_time)

    return b64jpgdataMajor, b64jpgdataMinor, b64jpgdataextraction1, b64jpgdatarejection1
