"""
Vidale Method.

:copyright:
    Eduardo Rodrigues de Almeida
:license:
    The MIT License (MIT)
    Copyright (c) 2021 MrEdwards
"""
import numpy as np
import scipy.signal as signal
import math 
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time
start_time = time.time()
import io
import base64


def vidale(data, window_size):
    """
    Obtaining the elliptical component of polarization, strike, inclination (dip), \
    polarization strength of the signal and the degree of planar polarization by implementing \
    the method designed by John E. Vidale. \
    Signal in Z, R, T orientation.

    :parameter data: Three component signal data.
    :type data: array   
    :parameter window_size: Size for the window.
    :type data: int
    :return: numpy array with elliptical component of polarization, numpy array with strike, \
    numpy array with dip, numpy array with polarization strenght of the signal and a numpy \
    array with degree of planar polarization.

    """
    sig = data

    # Application of the Hilbert transform
    t_h = signal.hilbert(sig[0])
    r_h = signal.hilbert(sig[1])
    z_h = signal.hilbert(sig[2])

    cov_matrix = np.zeros([3, 3], dtype=np.complex128)

    # Creation of the gaussian window
    window = signal.windows.gaussian(window_size, 4)
    signal_window_size = len(r_h)-len(window)
    
    elliptical_pol = np.zeros((signal_window_size))
    strike = np.zeros((signal_window_size))
    dip = np.zeros((signal_window_size))
    pol_strength = np.zeros((signal_window_size))
    degree_planar_pol = np.zeros((signal_window_size))
    for i in range(signal_window_size):
        
        # Windowing the signal
        z_w_gauss = window * z_h[i:i+(len(window))]
        t_w_gauss = window * t_h[i:i+(len(window))]
        r_w_gauss = window * r_h[i:i+(len(window))]
        
        t_w_gauss -= t_w_gauss.mean()
        r_w_gauss -= r_w_gauss.mean()
        z_w_gauss -= z_w_gauss.mean()

        # Building the covariance matrix
        cov_matrix = np.cov([z_w_gauss, r_w_gauss, t_w_gauss], bias=True)
        
        # Obtaining the eigenvalues and eigenvectors
        eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
        
        # Maximization of the lenght of the real component of the eigenvector
        def length_X_fun(alpha):
            return 1. - math.sqrt(
                pow((eig_vectors[0][2] * (math.cos(alpha) + math.sin(alpha) * 1j)).real, 2) +
                pow((eig_vectors[1][2] * (math.cos(alpha) + math.sin(alpha) * 1j)).real, 2) +
                pow((eig_vectors[2][2] * (math.cos(alpha) + math.sin(alpha) * 1j)).real, 2))

        lenght_eig_component = optimize.fminbound(length_X_fun, 0.0, 180.0, full_output=True)
        X = 1. - lenght_eig_component[1] #value of dependent variable which is length x[1]
        
        # ellliptical component
        elliptical_pol[i] = math.sqrt(1.0 - pow(X, 2)) / X
        
        # Strike and dip -- 0 ° strike and dip represents a vector which points horizontally in the direction back to the epicenter
        strike[i] = math.degrees(math.atan((eig_vectors[1][2]).real / (eig_vectors[0][2]).real ))
        dip[i] = math.atan((eig_vectors[2][2]).real / math.sqrt(pow((eig_vectors[0][2]).real, 2) 
        + pow((eig_vectors[1][2]).real, 2)))
        
        # Polarization strenght of the signal -- Ps is near 1 if the signal is completely polarized in that there is only primarily one 
        # component of polarization, but Ps is 0 if the largest component of polarization is only as big as the other two combined.
        pol_strength[i] = 1. - (eig_values[1] + eig_values[0] / eig_values[2])
        
        # Degree of planar polarization -- Pp is 1 if the intermediate component of polarization is much larger than the smallest component, 
        # but Pp is near 0 if the intermediate and smallest components of polarization are comparable.
        degree_planar_pol[i] = 1. - (eig_values[1] / eig_values[0])


    fig, axs = plt.subplots(3, 1)

    plt.sca(axs[0])
    plt.plot(dip, color='k', linewidth=1.5, label='dip')
    plt.title("Dip")
    plt.xlabel('Time (s)')
    #plt.ylabel('0 ° dip - vector points horizontally in the direction back to the epicenter.')

    plt.sca(axs[1])
    plt.plot(strike, color='k', linewidth=1.5, label='strike')
    plt.title("Strike")
    plt.xlabel('Time (s)')
    #plt.ylabel('0 ° strike - vector points horizontally in the direction back to the epicenter')

    plt.sca(axs[2])
    plt.plot(elliptical_pol, color='k', linewidth=1.5, label='Pe')
    plt.title("Elliptical component of polarization (Pe)")
    plt.xlabel('Time (s)')
    #plt.ylabel('1 and 0 for circularly/linearly polarized motion')

    StringIObytes1 = io.BytesIO()
    fig.tight_layout()
    plt.savefig(StringIObytes1, format='jpg')

    StringIObytes1.seek(0)
    b64jpgdata1 = base64.b64encode(StringIObytes1.read()).decode()
    plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 1)

    plt.sca(axs[0])
    plt.plot(pol_strength, color='k', linewidth=1.5, label='Ps')
    plt.title("Polarization strenght of the signal (Ps)")
    plt.xlabel('Time (s)')
    #plt.ylabel('Ps is near 1 = signal is completely polarized, Ps = 0 the largest component of polarization is only as big as the other two combined.')

    plt.sca(axs[1])
    plt.plot(degree_planar_pol, color='k', linewidth=1.5, label='Pp')
    plt.title("Degree of planar polarization (Pp)")
    plt.xlabel('Time (s)')
    #plt.ylabel('Pp = 1 - the intermediate component of polarization is much larger than the smallest component, Pp =/- 0 - the intermediate and smallest components of polarization are comparable.')

    StringIObytes2 = io.BytesIO()
    fig.tight_layout()
    plt.savefig(StringIObytes2, format='jpg')
    StringIObytes2.seek(0)
    b64jpgdata2 = base64.b64encode(StringIObytes2.read()).decode()
    plt.show()
    plt.close()
    
    print("Execution time:", time.time()-start_time)

    return b64jpgdata1, b64jpgdata2 #elliptical_pol, strike, dip, pol_strength, degree_planar_pol, 
