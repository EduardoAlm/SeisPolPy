import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import time
start_time = time.time()
import io
import base64
import mat4py

class FlinnMethod:
    """


    :param data: Three component signal data.
    :type data: array   
    """

    def __init__(self, data):
        self.data = data

    def flinn(self):
        """
        Obtaining rectilinearity and direction of particle motion by implementing the Flinn method.

        
        """
        sig = self.data.obj

        plt.rcParams['figure.figsize'] = [16, 12]
        plt.rcParams.update({'font.size': 18})

        #t = np.transpose(sig[0])  # T component vector
        #r = np.transpose(sig[1])  # R component vector
        #z = np.transpose(sig[2])  # Z component vector
        t = sig[0] # T component vector
        r = sig[1]  # R component vector
        z = sig[2]  # Z component vector
        # Adding noise to data
        #noise = np.random.normal(0, 0.0000000, sig.shape)
        #noisy_signal = sig + noise  # Adding noise vector data to our signal vector
        #t = noisy_signal[0]
        #r = noisy_signal[1]
        #z = noisy_signal[2]

        # Windowing the data
        window_size = 50
       # t_windowed = FlinnMethod.rolling_window(self, x1, window_size)
       # r_windowed = FlinnMethod.rolling_window(self, x2, window_size)
       # z_windowed = FlinnMethod.rolling_window(self, x3, window_size)

        window = signal.windows.gaussian(window_size, 4)

       # t_w_gauss = t_windowed * window
       # r_w_gauss = r_windowed * window
       # z_w_gauss = z_windowed * window
        signal_window_size = len(r)-len(window)
        # Obtaining L1, L2, L3 principal axis
        g1 = np.zeros((signal_window_size, 1))
        g2_r = np.zeros((signal_window_size, 1))
        g2_t = np.zeros((signal_window_size, 1))
        g2_z = np.zeros((signal_window_size, 1))
        
        #alpha=np.pi
        #inc=(0/180)*np.pi
        d_z=np.array([1, 0, 0])
        d_r=np.array([0, 1, 0])
        d_t=np.array([0, 0, 1])
       # d_t=np.array([1, 0, 0])
       # d_r=np.array([0, 1, 0])
       # d_z=np.array([0, 0, 1])
        
        eigarr_1 = np.zeros((signal_window_size))
        eigarr_2 = np.zeros((signal_window_size))
        eigarr_3 = np.zeros((signal_window_size))
        for i in range(1, signal_window_size):
            z_w_gauss = window * z[i:i+(len(window))]
            t_w_gauss = window * t[i:i+(len(window))]
            r_w_gauss = window * r[i:i+(len(window))]
            #transpose_arr = np.transpose(np.array([t_w_gauss, r_w_gauss, z_w_gauss]))
            arr = np.array([z_w_gauss, r_w_gauss, t_w_gauss])
           
            cov_matrix = np.cov(arr, bias=True)  # convolutional matrix
            eig_values, v = np.linalg.eigh(cov_matrix)
         
            eigarr_1[i] = eig_values[0]
            eigarr_2[i] = eig_values[1]
            eigarr_3[i] = eig_values[2]
          
            g1[i] =  (1 - np.sqrt(eig_values[1]/eig_values[2]))
            g2_z[i] = np.dot(v[2], np.transpose(d_z))  # Z
            g2_r[i] = np.dot(v[2], np.transpose(d_r))  # R
            g2_t[i] = np.dot(v[2], np.transpose(d_t))  # T
            
            

        g1 = np.vstack((np.zeros((len(window), 1)), g1))
        g2_z = np.vstack((np.zeros((len(window), 1)), g2_z)) 
        g2_r = np.vstack((np.zeros((len(window), 1)), g2_r))
        g2_t = np.vstack((np.zeros((len(window), 1)), g2_t))
       

        
        fig, axs = plt.subplots(3, 1)
        plt.sca(axs[0])
        plt.plot(eigarr_1, color='r', linewidth=1.5, label='')
        plt.title("eigvalues 1")
        plt.xlabel('Time (s)')
        plt.ylabel('eig')
        plt.sca(axs[1])
        plt.plot(eigarr_2, color='r', linewidth=1.5, label='')
        plt.title("eigvalues 2")
        plt.xlabel('Time (s)')
        plt.ylabel('eig')

        plt.sca(axs[2])
        plt.plot(eigarr_3, color='r', linewidth=1.5, label='')
        plt.title("eigvalues 3")
        plt.xlabel('Time (s)')
        plt.ylabel('eig')
        fig.tight_layout()
    
        fig, axs = plt.subplots(3, 1)
        plt.sca(axs[0])
        plt.plot(z, color='r', linewidth=1.5, label='z')
        plt.title("z - original")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.sca(axs[1])
        plt.plot(r, color='c', linewidth=1.5, label='r')
        plt.title("r - original")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.sca(axs[2])
        plt.plot(t, color='k', linewidth=1.5, label='t')
        plt.title("t - original")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        fig.tight_layout()

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
        
        plt.show()
        
        StringIObytes.seek(0)
        b64jpgdata = base64.b64encode(StringIObytes.read())

        print("Execution time:", time.time()-start_time)

        return b64jpgdata

data = mat4py.loadmat('tests/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
res = FlinnMethod.flinn(sig)
print(res)