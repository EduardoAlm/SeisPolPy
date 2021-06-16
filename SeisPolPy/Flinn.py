"""
Flinn Method.

:copyright:
    Eduardo Rodrigues de Almeida
:license:
    The MIT License (MIT)
    Copyright (c) 2021 MrEdwards
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
start_time = time.time()
import io
import base64


def flinn(data, window_size):
    """
    Obtaining rectilinearity and direction of particle motion in a three component signal by \
    implementing the method designed by E. A. Flinn. \
    Signal in Z, R, T orientation. 

    :parameter data: Three component signal data.
    :type data: array   
    :parameter window_size: Size for the window.
    :type data: int
    :return: array with rectilinearity values, array with polarization direction of the z \
    component, array with polarization direction of the r component, array with polarization \
    direction of the t component and a base64 encoded string of bytes containing the previous \
    arrays plots.

    """
    sig = data

    plt.rcParams['figure.figsize'] = [16, 12]
    plt.rcParams.update({'font.size': 18})

    t = sig[0] # T component vector
    r = sig[1]  # R component vector
    z = sig[2]  # Z component vector
    
    # Creation of the gaussian window
    window = signal.windows.gaussian(window_size, 4)

    signal_window_size = len(r)-len(window)

    g1 = np.zeros((signal_window_size, 1))
    g2_r = np.zeros((signal_window_size, 1))
    g2_t = np.zeros((signal_window_size, 1))
    g2_z = np.zeros((signal_window_size, 1))
    
    d_z=np.array([1, 0, 0])
    d_r=np.array([0, 1, 0])
    d_t=np.array([0, 0, 1])

    for i in range(1, signal_window_size):
        # Windowing the signal
        z_w_gauss = window * z[i:i+(len(window))]
        t_w_gauss = window * t[i:i+(len(window))]
        r_w_gauss = window * r[i:i+(len(window))]

        t_w_gauss -= t_w_gauss.mean()
        r_w_gauss -= r_w_gauss.mean()
        z_w_gauss -= z_w_gauss.mean()
        
        arr = np.array([z_w_gauss, r_w_gauss, t_w_gauss])
        
        # Building the convolutional matrix
        cov_matrix = np.cov(arr, bias=True) 

        # Obtaining the eigenvalues and eigenvectors
        eig_values, v = np.linalg.eigh(cov_matrix)
        
        # Obtaining the rectilinearity for the window of the signal
        g1[i] =  (1 - np.sqrt(eig_values[1]/eig_values[2]))
        # Obtaining the direction of particle motion for the window 
        # of the signal diferent components 
        g2_z[i] = np.dot(v[2], np.transpose(d_z))  
        g2_r[i] = np.dot(v[2], np.transpose(d_r))  
        g2_t[i] = np.dot(v[2], np.transpose(d_t))  
        
    g1 = np.vstack((np.zeros((len(window), 1)), g1))
    g2_z = np.vstack((np.zeros((len(window), 1)), g2_z)) 
    g2_r = np.vstack((np.zeros((len(window), 1)), g2_r))
    g2_t = np.vstack((np.zeros((len(window), 1)), g2_t))

    fig, axs = plt.subplots(4, 1)
    plt.sca(axs[0])
    plt.plot(g1, color='c', linewidth=1.5, label='rec')
    plt.title("Rectilinearity")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.sca(axs[1])
    plt.plot(g2_z, color='r', linewidth=1.5, label='z')
    plt.title("DIRECTION OF PARTICLE MOTION z")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.sca(axs[2])
    plt.plot(g2_r, color='c', linewidth=1.5, label='r')
    plt.title("DIRECTION OF PARTICLE MOTION r")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.sca(axs[3])
    plt.plot(g2_t, color='k', linewidth=1.5, label='t')
    plt.title("DIRECTION OF PARTICLE MOTION t")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    StringIObytes = io.BytesIO()
    fig.tight_layout()
    plt.savefig(StringIObytes, format='jpg')
    StringIObytes.seek(0)
    b64jpgdata = base64.b64encode(StringIObytes.read()).decode()
    plt.close()
    print("Execution time:", time.time()-start_time)

    return b64jpgdata #g1, g2_z, g2_r, g2_t, b64jpgdata
