import numpy as np
import matplotlib.pyplot as plt
import time
import math
start_time = time.time()
import scipy.sparse as sp
import adjoint
DTYPE = np.float64
import diags
import io 
import base64

class PinnegarMethod:
    """
    """
    def __init__(self, data):
        self.data = data

    def forward(N, s):
        """
        """
        id = np.linspace(0, N-1, N, dtype="int32")
        w = np.zeros([N, N], dtype=DTYPE)

        if type(s) == int:
            sj = s * np.ones([N, 1]).astype(int)
        elif len(s) != N:
            print('length of "s" should be one or the same as length of data ')

        for i in range(N-1):
            w[:, i] = np.exp(np.multiply(-2 * pow(np.pi, 2), pow(id - i, 2)) / pow(sj[i], 2))

        offset = np.linspace(0, pow(N, 2) - N, N, dtype="int32")
        y = np.transpose(w)
        res = diags.diagonal(y.tolist(), offset,  N, pow(N, 2), dtype=DTYPE)
        return sp.dia_matrix((res, offset), shape=(N, pow(N, 2)))

    def stft(x, s):
        """
        """
        N = len(x)
        G = PinnegarMethod.forward(N, s)
        tfx = adjoint.adjoin(G, N, x)
        return np.reshape(tfx,(N, N))

    def pinnegar(self):
        """
        """
        sig = self.data.obj

        plt.rcParams['figure.figsize'] = [16, 12]
        plt.rcParams.update({'font.size': 18})
        showgraph = 0

        dt = 1.5
        s = 100

        length = len(sig[0])
        half = math.ceil(length / 2)
        half += 550 #best so far half + 350 = 900

        tfrx = PinnegarMethod.stft(sig[1][0:(half)], s)  # R component vector #, window=np.gaussian(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2] # 0:sample freq 1: segment times 2:STFT of t
        tfry = PinnegarMethod.stft(sig[0][0:(half)], s)  # T component vector #, window=np.hanning(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2]  # 0:sample freq 1: segment times 2:STFT of r
        tfrz = PinnegarMethod.stft(sig[2][0:(half)], s)  # Z component vector #, window=np.hanning(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2]  # 0:sample freq 1: segment times 2:STFT of z

        if showgraph == 1:
            fig, axs = plt.subplots(3, 1)
            plt.sca(axs[0])
            plt.plot(tfrx, color='c', linewidth=0.5, label='T')
            plt.title("T component - STFT")
            plt.xlabel('Time (s)')
            plt.ylabel('Freq (kHz)')

            plt.sca(axs[1])
            plt.plot(tfry, color='k', linewidth=0.5, label='R')
            plt.title("R component - STFT")
            plt.xlabel('Time (s)')
            plt.ylabel('Freq (kHz)')

            plt.sca(axs[2])
            plt.plot(tfrz, color='r', linewidth=0.5, label='Z')
            plt.title("Z component - STFT")
            plt.xlabel('Time (s)')
            plt.ylabel('Freq (kHz)')

            fig.tight_layout()
            plt.savefig("./results/pinnegarSFTFofTRZ.png")
            plt.show()

        sminor = np.zeros((len(tfrx), half))
        smajor = np.zeros((len(tfrx), half))
        inclin = np.zeros((len(tfrx), half))  # inclination
        omega = np.zeros((len(tfrx), half))  # pitch (the angle between the ascending node and the position of maximum displacement)
        phi = np.zeros((len(tfrx), half))  # phase
        ohm = np.zeros((len(tfrx), half))  # strike (the azimuth of the ascending node)
        omega0 = np.zeros((len(tfrx), half))
        phi0 = np.zeros((len(tfrx), half))


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

        plt.matshow(np.abs(np.transpose(smajor[:][:,:])))
        plt.title("Semi Major")
        
        plt.matshow(np.abs(np.transpose(sminor[:][:,:])))
        plt.title("Semi Minor")

        StringIObytes = io.BytesIO()
        fig.tight_layout()
        plt.savefig(StringIObytes, format='jpg')
        plt.show()

        StringIObytes.seek(0)
        b64jpgdata = base64.b64encode(StringIObytes.read())

        print("Time of execution", time.time()-start_time)

        return smajor, sminor, inclin, omega, phi, ohm, omega0, phi0, b64jpgdata

