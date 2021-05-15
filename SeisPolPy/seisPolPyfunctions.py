import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.sparse as sp
import math

def stft(x, s):
    N = len(x)
    G = forw(N, s)
    At = (lambda e1: np.reshape(
        np.fft.fft(np.reshape(np.matmul(np.transpose(G),e1[:]), (N, N))),
                                (pow(N, 2), 1)))  #ad joint
    tfx = At(x)
    tfx = np.reshape(tfx,(N, N))
    return tfx

def forw(N, s):
    id = np.linspace(0, N-1, N).astype(int)
    w = np.zeros([N, N])

    if type(s) == int:
        s = s * np.ones([N, 1]).astype(int)
    elif len(s) != N:
        raise Exception('length of "s" should be one or the same as length of data ')

    for i in range(N-1):
        w[:, i] = np.exp(np.multiply(-2 * pow(np.pi, 2), pow(id - i, 2)) / pow(s[i], 2))

    diagonal = np.linspace(0, pow(N, 2) - N, N, dtype="uint32")
    y = np.transpose(w)
    z = (N, pow(N, 2))
    results = sp.diags(y, diagonal, z).toarray()

    return results

def pinnegar(data: dict, dt: float, s: int):

    sig = np.array([data['t'], data['r'], data['z']])
    length = len(sig[0])
    half = math.ceil(length / 2)
    half += 550 #best so far half + 350 = 900

    tfrx = stft(sig[1][0:(half)], s)  # R component vector #, window=np.gaussian(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2] # 0:sample freq 1: segment times 2:STFT of t
    tfry = stft(sig[0][0:(half)], s)  # T component vector #, window=np.hanning(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2]  # 0:sample freq 1: segment times 2:STFT of r
    tfrz = stft(sig[2][0:(half)], s)  # Z component vector #, window=np.hanning(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2]  # 0:sample freq 1: segment times 2:STFT of z

    sminor = np.zeros((len(tfrx), half))
    smajor = np.zeros((len(tfrx), half))
    inclin = np.zeros((len(tfrx), half))  # inclination
    omega = np.zeros((len(tfrx), half))  # pitch (the angle between the ascending node and the position of maximum displacement)
    phi = np.zeros((len(tfrx), half))  # phase
    ohm = np.zeros((len(tfrx), half))  # strike (the azimuth of the ascending node)
    omega0 = np.zeros((len(tfrx), half))
    phi0 = np.zeros((len(tfrx), half))
    print(len(tfrx))

    for i in range(half):
        X = tfrx[:,i]
        Y = tfry[:,i]
        Z = tfrz[:,i]

        XR = np.real(X)
        YR = np.real(Y)
        ZR = np.real(Z)

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
        b = np.multiply((1 / np.sqrt(2)), (np.sqrt(A - D)))
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
        #print(np.sqrt(np.size(sminor)))
        #print(np.size(a))

        smajor[:, i] = a
        sminor[:, i] = b
        inclin[:, i] = I  # inclination
        omega[:, i] = w  # pitch (the angle between the ascending node and the position of maximum displacement)
        phi[:, i] = ph  # phase
        ohm[:, i] = OHM  # strike (the azimuth of the ascending node)
        omega0[:, i] = w0
        phi0[:, i] = ph0

    plt.matshow(np.abs(np.transpose(smajor[:][:,0:549])))
    plt.figure()
    plt.show()
    plt.matshow(np.abs(np.transpose(sminor[:][:,0:549])))
    plt.figure()
    plt.show()

    return smajor, sminor

def rolling_window(a, window):
    # Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size), window_size, x))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rectilinear(L1, L2):

    if (1 - (L2/L1)) < 1 and (1 - (L2/L1)) > 0:
        return (1 - (L2 / L1))
    else:
        raise Exception("Not in interval")

def flinn(data: dict):

    sig = np.array([data['t'], data['r'], data['z']])

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
    fig.tight_layout()
    plt.show()
    return g1, g2_r, g2_z, g2_t



# def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
 #   """
 #   Calculate the great circle distance between two points on the
 #   earth (specified in decimal degrees), returns the distance in
 #   meters.    All arguments must be of equal length.    :param lon1: longitude of first place
 #   :param lat1: latitude of first place
 #   :param lon2: longitude of second place
 #   :param lat2: latitude of second place
 #   :return: distance in meters between the two sets of coordinates
 #   """
    # Convert decimal degrees to radians
 #   lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2,
 #   lat2])    # Haversine formula
 #   dlon = lon2 - lon1
 #   dlat = lat2 - lat1
 #   a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
 #   c = 2 * np.arcsin(np.sqrt(a))
 #   km = 6367 * c
 #   return km * 1000