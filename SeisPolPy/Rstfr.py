"""
RSTFR Method.

:copyright:
    Eduardo Rodrigues de Almeida
:license:
    The MIT License (MIT)
    Copyright (c) 2021 MrEdwards
"""
from numpy import abs, int16, maximum, conj, ceil, flipud, add, concatenate, reshape, int32, zeros, array, real, sqrt, dot, multiply, linspace, exp, float64, ones, pi, transpose, float16
from numpy.linalg import eigh
from numpy import abs as abso
from numpy import max as maxim
import matplotlib.pyplot as plt
import time
import base64
import io
start_time = time.time()
from scipy.sparse import dia_matrix
import forw_op  
import adjoint 
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
    tfx, tfy, tfz = zeros([pow(N, 2), 1], dtype=float), zeros([pow(N, 2), 1], dtype=float), zeros([pow(N, 2), 1], dtype=float)
   
    
    G = forward(N, s)
    
    lamb_da = multiply(multiply(mu, abso(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
    i=0
    while i < n_it:
        U1j = tfx + multiply(mu, adjoint.adjoin(G, N, (d1 - forw_op.forw_o(G, N, tfx))))
        U2j = tfy + multiply(mu, adjoint.adjoin(G, N, (d2 - forw_op.forw_o(G, N, tfy))))
        U3j = tfz + multiply(mu, adjoint.adjoin(G, N, (d3 - forw_op.forw_o(G, N, tfz))))
        tfx = soft_threshholding(U1j, lamb_da)
        tfy = soft_threshholding(U2j, lamb_da)
        tfz = soft_threshholding(U3j, lamb_da)
        d1 = d1 + (dh1 - real(forw_op.forw_o(G, N, tfx)))
        d2 = d2 + (dh2 - real(forw_op.forw_o(G, N, tfy)))
        d3 = d3 + (dh3 - real(forw_op.forw_o(G, N, tfz)))
        i+=1
    
    return reshape(tfx, (N, N)), reshape(tfy, (N, N)), reshape(tfz, (N, N))

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
    :return: array containing the obtained directivity values. 
    """
    
    amp_attr = sqrt(2)*eig_values[2]/L
    if eig_values[2] >= 0 and eig_values[2] < zeta:
        ampli = 0.
    elif eig_values[2] > zeta and eig_values[2] < eta:
        ampli = cos(pi*(amp_attr-zeta)/2*(eta-zeta))
    elif eig_values[2] > eta and eig_values[2] <= 1:
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
    b_vec=array([[1,0,0],[0,1,0],[0,0,1]], int)

    degree_res = maxim(abso(dot(eig_vec3[2], b_vec[0]))) 
    degree_dir = float("{:.2f}".format(round(degree_res, 2)))      
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
    degree_dir = float("{:.2f}".format(round(degree_res, 2)))
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
    :return: array containing the obtained rectilinearity values. 
    """
    print("dir_rec")
    degree_res = 1-((eig_values[1]+eig_values[0])/eig_values[2])
    degree_rec = float("{:.2f}".format(round(degree_res, 2)))
    if degree_rec >= -1 and degree_rec < alpha:
        rec = 1.
    elif degree_rec > alpha and degree_rec < beta:
        rec = cos(pi*(degree_rec-alpha)/2*(beta-alpha))
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

    sig, start_time = data, time.time()

    plt.rcParams['figure.figsize'] = [13, 8]
    plt.rcParams.update({'font.size': 18})

    t, r, z = sig[0][0:(len(sig[0]))], sig[1][0:(len(sig[0]))], sig[2][0:(len(sig[0]))]

    N, mu = len(r), 1e-3

    if alg == "stft":
        # Obtaining the Short Time Fourier Transform for each component
        tfrx, tfry, tfrz = stft(r, s),  stft(t, s), stft(z, s) # R component vector 
  
    elif alg == "s_stft":
        # Obtaining the sparse Short Time Fourier Transform for each component
        tfrx, tfry, tfrz = stft_s_ist(r, t, z, s, n_it, mu)  
    else:
        raise Exception("Please choose an available method (stft/s_stft).")

    nf = ceil(N/2).astype(int)
    semi, majo, mino, majon, minon,i  = zeros((12,nf,N)), zeros((3,nf,N)), zeros((3,nf,N)), \
        zeros((3,nf,N)), zeros((3,nf,N)), 0
    
    while i < N:
        # Generates the semi major/minor
        semi[:,:,i], majo[:,:,i], mino[:,:,i], majon[:,:,i], minon[:,:,i] = semimm(multiply(10, tfry[:,i]),\
            multiply(10, tfrx[:,i]), multiply(10, tfrz[:,i]))
        i+=1
        print("Generating the semi major/minor ... ",np.round(i*100/N, 2), "%")

    n2, n = len(semi[0, 0]), len(semi[0])
   
    
    majornorm, minornorm, i, j= zeros((nf, N)), zeros((nf, N)), 0, 0
     
    while i < nf:
        while j < N:
            
            # Normalizes the semi major and minor values
            majornorm[i, j], minornorm[i, j] = sqrt(dot(majo[:, i, j],majo[:, i, j])), sqrt(dot(mino[:, i, j],mino[:, i, j]))
            j += 1
        i+=1
        print("Normalising the output values ... ",round(i*100/nf, 2), "%")

    cax = maxim((maxim(majornorm), maxim(minornorm)))
    
    plt.imshow(abso(majornorm), cmap='hot', alpha=1, vmin=0, vmax=0.7*cax)
    plt.title('RS-TFR (SM)')    
   
    StringIObytes1 = io.BytesIO()
    plt.savefig(StringIObytes1, format='jpg')

    StringIObytes1.seek(0)
    b64jpgdataMajor = base64.b64encode(StringIObytes1.read()).decode()
    plt.close()
    plt.imshow(abso(minornorm), cmap="hot", alpha=1, vmin=0, vmax=0.7*cax)
    plt.title('RS-TFR (Sm)')
   
    StringIObytes2 = io.BytesIO()
    plt.savefig(StringIObytes2, format='jpg')

    StringIObytes2.seek(0)
    b64jpgdataMinor = base64.b64encode(StringIObytes2.read()).decode()
    plt.close()

    rec_filter, amp_filter, dir_love_filter, dir_rayleigh_filter, i, j= zeros((n2, n2), dtype=float16), zeros((n2, n2), dtype=float16), zeros((n2,n2), dtype=float16), zeros((n2, n2), dtype=float16), 0, 0
 
    while i < n:
        while j < n2:
            eig_values = array([semi[9][0,0],semi[10][0,0],semi[11][0,0]])
            biggesteig_vec=array([semi[6][0,0],semi[7][0,0], semi[8][0,0]])        
            rec_filter[j, j] = rectilinearity(alpha, beta, eig_values)
            amp_filter[j, j] = amplitude(zeta, eta, eig_values, N)

            if filt=="love":
                dir_love_filter[j, j] = directivity_love(gamma, lamb_da, biggesteig_vec)
            elif filt=="rayleigh":
                dir_rayleigh_filter[j, j] = directivity_rayleigh(gamma, lamb_da, biggesteig_vec)
            else:
                raise Exception("The chosen options is not available please check the documentation.")
            j += 1
        i+=1
    if filt=="love":
        rej_love_filter = 1 - multiply((1 - rec_filter), (1 - dir_love_filter), (1 - amp_filter))
        ext_love_filter = (1 - rec_filter)
        rejected_x = dot(tfrx, rej_love_filter)
        rejected_y = dot(tfry, rej_love_filter)
        rejected_z = dot(tfrz, rej_love_filter)
        extracted_x = dot(tfrx, ext_love_filter)
        extracted_y = dot(tfry, ext_love_filter)
        extracted_z = dot(tfrz, ext_love_filter)
    elif filt=="rayleigh":
        rej_rayleigh_filter = 1 - multiply((1 - rec_filter), (1 - dir_rayleigh_filter), (1 - amp_filter))
        ext_rayleigh_filter = (1 - rec_filter)
        rejected_x = multiply(tfrx, rej_rayleigh_filter)
        rejected_y = multiply(tfry, rej_rayleigh_filter)
        rejected_z = multiply(tfrz, rej_rayleigh_filter)
        extracted_x = multiply(tfrx, ext_rayleigh_filter)
        extracted_y = multiply(tfry, ext_rayleigh_filter)
        extracted_x = multiply(tfrz, ext_rayleigh_filter)
    else:
        raise Exception("The chosen options is not available please check the documentation.")            
    
    
    plt.figure()

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
    
    plt.close()


    print("Execution time:", time.time()-start_time)

    return b64jpgdataMajor, b64jpgdataMinor, b64jpgdataextraction, b64jpgdatarejection
