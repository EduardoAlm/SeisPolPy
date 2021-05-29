from rectilinearity_fun import rectilinear
from window_extraction import rolling_window
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
start_time = time.time()
import io
import base64

class FlinnMethod:
    """


    :param data: Three component signal data.
    :type data: array   
    """

    def __init__(self, data):
        self.data = data

    
    def rolling_window(self, a, window):
        """
        Reshapes a numpy array 'a' of shape (n, x) to form shape((n - window_size), window_size, x)).

        :param a: One component signal data.
        :type a: array

        :param window: Gaussian Window.
        :type window: array

        :return: Windowed signal.
        :rtype:
        """
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def rectilinear(self, L1, L2):
        """
        Checks if the obtained rectilinearity value is between the interval 0 and 1.

        :param L1: The eigenvalue.
        :type L1: float

        :param L2: The eigenvector.
        :type L2: float array

        :return: The rectilinearity value.
        :rtype: float

        """
        if (1 - (L2/L1)) < 1 and (1 - (L2/L1)) > 0:
            return (1 - (L2/L1))
        else:
            raise Exception("Not in interval")

    def flinn(self):
        """
        Obtaining rectilinearity and direction of particle motion by implementing the Flinn method.

        
        """
        sig = self.sig

        show_window_plot = input()
        plt.rcParams['figure.figsize'] = [16, 12]
        plt.rcParams.update({'font.size': 18})

        L = len(sig[0])  # Length of each component
        x1 = sig[0]  # T component vector
        x2 = sig[1]  # R component vector
        x3 = sig[2]  # Z component vector

        # Adding noise to data
        noise = np.random.normal(0, 0.00001, sig.shape)
        noisy_signal = sig + noise  # Adding noise vector data to our signal vector
        t = noisy_signal[0]
        r = noisy_signal[1]
        z = noisy_signal[2]


        # Data Plotting
        fig, axs = plt.subplots(3, 1)
        plt.sca(axs[0])
        plt.plot(t, color='c', linewidth=1.5, label='T')
        plt.title("T component - Noisy Form")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.sca(axs[1])
        plt.plot(r, color='k', linewidth=1.5, label='R')
        plt.title("R component - Noisy Form")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.sca(axs[2])
        plt.plot(z, color='r', linewidth=1.5, label='Z')
        plt.title("Z component - Noisy Form")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        fig.tight_layout()
        plt.savefig("results/noisyflinn.png")
        plt.show()

        # Windowing the data
        window_size = 200
        t_windowed = rolling_window(x1, window_size)
        r_windowed = rolling_window(x2, window_size)
        z_windowed = rolling_window(x3, window_size)

        window = signal.windows.gaussian(window_size, 4)

        t_w_gauss = t_windowed * window
        r_w_gauss = r_windowed * window
        z_w_gauss = z_windowed * window

        alpha=np.pi
        inc=(0/180)*np.pi
        d_z=np.array([np.cos(inc), np.sin(inc)*np.sin(alpha), np.sin(inc)*np.cos(alpha)])
        d_r=np.array([np.sin(inc), np.cos(inc)*np.sin(alpha), np.cos(inc)*np.cos(alpha)])
        d_t=np.array([0, np.cos(alpha), np.sin(alpha)])

        signal_window_size = len(t_windowed)
        # Obtaining L1, L2, L3 principal axis
        g1 = np.zeros((signal_window_size, 1))
        g2_r = np.zeros((signal_window_size, 1))
        g2_t = np.zeros((signal_window_size, 1))
        g2_z = np.zeros((signal_window_size, 1))

        for i in range(1,signal_window_size):

            cov_matrix = np.cov(np.array([z_w_gauss[i], t_w_gauss[i], r_w_gauss[i]]))  # convolutional matrix
            eig_values, eig_vectors = np.linalg.eig(cov_matrix)

            g1[i] = rectilinear(eig_values[0], eig_values[1])
            g2_r[i] = np.matmul(np.transpose(eig_vectors[0]), np.transpose(d_r))  # R
            g2_t[i] = np.matmul(np.transpose(eig_vectors[0]), np.transpose(d_t))  # T
            g2_z[i] = np.matmul(np.transpose(eig_vectors[0]), np.transpose(d_z))  # Z

        g1 = np.vstack((np.zeros((len(window), 1)), g1))
        g2_r = np.vstack((np.zeros((len(window), 1)), g2_r))
        g2_t = np.vstack((np.zeros((len(window), 1)), g2_t))
        g2_z = np.vstack((np.zeros((len(window), 1)), g2_z))

        fig, axs = plt.subplots(4, 1)
        plt.sca(axs[0])
        plt.plot(g1, color='c', linewidth=1.5, label='T')
        plt.title("Rectilinearity")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.sca(axs[1])
        plt.plot(g2_z, color='r', linewidth=1.5, label='Z')
        plt.title("DIRECTION OF PARTICLE MOTION Z")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.sca(axs[2])
        plt.plot(g2_r, color='c', linewidth=1.5, label='T')
        plt.title("DIRECTION OF PARTICLE MOTION R")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.sca(axs[3])
        plt.plot(g2_t, color='k', linewidth=1.5, label='R')
        plt.title("DIRECTION OF PARTICLE MOTION T")
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

