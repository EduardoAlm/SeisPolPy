import numpy as np
import matplotlib.pyplot as plt
import time
import mat4py
import base64
import io
start_time = time.time()
import scipy.sparse as sp
import forw_op
import adjoint
import math
DTYPE = np.float64
import diags


class RSTFRMethod:
    """
    """
    def __init__(self, data):
        self.data = data

    def filtering():
        """
        """
        return 1

    def soft_threshholding(z, T):
        """
        """
        az = np.abs(z)
        res = np.maximum(az-T, 0)/(np.maximum(az-T,0)+T)*z
        return np.abs(res)

    def cross(x1, x2):
        """
        """
        x = x1 * np.conj(x2)
        length = len(x1)
        half = np.ceil(length/2).astype(np.int)
        fstpart = x[1:(half)]
        sndpart = np.flipud(x[(half+1):])
        sumofparts = np.add(fstpart, sndpart)
        corr = np.concatenate(([x[0]], sumofparts))/length
        return corr

    def semimm(t, r, z):
        """
        """
        length = len(t)
        half = math.ceil(length/2)
        semi = np.zeros((12, half))
        major = np.zeros((3, half))
        minor = np.zeros((3, half))
        tt = RSTFRMethod.cross(t, t)
        rr = RSTFRMethod.cross(r, r)
        zz = RSTFRMethod.cross(z, z)
        tr = RSTFRMethod.cross(t, r)
        tz = RSTFRMethod.cross(t, z)
        rz = RSTFRMethod.cross(r, z)
        for i in range(half):
            cov_matrix = np.array([[tt[i], tr[i], tz[i]],[tr[i], rr[i], rz[i]],[tz[i],rz[i], zz[i]]], dtype=float)
            eig_values, V = np.linalg.eigh(cov_matrix)
            V = np.transpose(V)
            semi[0:3, i] = V[:, 2]
            semi[3:6, i] = V[:, 1]
            semi[6:9, i] = V[:, 0]
            semi[9:, i] = eig_values
            facc = np.sqrt(length)/np.sqrt(2)
            major[:, i] = np.dot(np.dot(facc, np.sqrt(eig_values[2])), V[:, 2])
            minor[:, i] = np.dot(np.dot(facc, np.sqrt(eig_values[1])), V[:, 1])
        major_norm = major/np.max(np.abs(major))
        minor_norm = minor/np.max(np.abs(minor))
        return semi, major, minor, major_norm, minor_norm

    def forward(N, s):
        """
        """
        #cdef np.ndarray
        id = np.linspace(0, N-1, N, dtype="int32")
        #cdef np.ndarray
        w = np.zeros([N, N], dtype=DTYPE)
        #cdef np.ndarray sj
        #cdef int i

        if type(s) == int:
            sj = s * np.ones([N, 1]).astype(int)
        elif len(s) != N:
            print('length of "s" should be one or the same as length of data ')

        for i in range(N-1):
            w[:, i] = np.exp(np.multiply(-2 * pow(np.pi, 2), pow(id - i, 2)) / pow(sj[i], 2))

        #cdef np.ndarray
        offset = np.linspace(0, pow(N, 2) - N, N, dtype="int32")
        #cdef np.ndarray
        y = np.transpose(w)

        res = diags.diagonal(y.tolist(), offset,  N, pow(N, 2), dtype=DTYPE)
        return sp.dia_matrix((res, offset), shape=(N, pow(N, 2)))

    def stft_s_ist(x, y, z, s, n_it, mu):
        """
        """
        N = len(x)
        d1, d2, d3 = x, y, z
        dh1, dh2, dh3 = d1, d2, d3
        tfx, tfy, tfz = np.zeros([pow(N, 2), 1], dtype=float), np.zeros([pow(N, 2), 1], dtype=float), np.zeros([pow(N, 2), 1], dtype=float)

        G = RSTFRMethod.forward(N, s)
        
        lamb_da = np.multiply(np.multiply(mu, np.abs(adjoint.adjoin(G, N, d1)).max(0)), 1.2) 
        
        for i in range(n_it):
            U1j = tfx + np.multiply(mu, adjoint.adjoin(G, N, (d1 - forw_op.forw_o(G, N, tfx))))
            U2j = tfy + np.multiply(mu, adjoint.adjoin(G, N, (d2 - forw_op.forw_o(G, N, tfy))))
            U3j = tfz + np.multiply(mu, adjoint.adjoin(G, N, (d3 - forw_op.forw_o(G, N, tfz))))
            tfx = RSTFRMethod.soft_threshholding(U1j, lamb_da)
            tfy = RSTFRMethod.soft_threshholding(U2j, lamb_da)
            tfz = RSTFRMethod.soft_threshholding(U3j, lamb_da)
            d1 = d1 + (dh1 - np.real(forw_op.forw_o(G, N, tfx)))
            d2 = d2 + (dh2 - np.real(forw_op.forw_o(G, N, tfy)))
            d3 = d3 + (dh3 - np.real(forw_op.forw_o(G, N, tfz)))
            print("Forward/Backward splitting and threshholding... ", np.round(i*100/N, 2), "%")
        
        return np.reshape(tfx, (N, N)), np.reshape(tfy, (N, N)), np.reshape(tfz, (N, N))

    def stft(x, s):
        """
        """
        N = len(x)
        G = RSTFRMethod.forward(N, s)

        tfx = adjoint.adjoin(G, N, x)

        return np.reshape(tfx,(N, N))


    def rstfr(self):
        """
        """
        sig = self.data.obj
        start_time = time.time()
        input_var=input("Do you which to run the code using normal STFT or using the sparse STFT: [1. STFT | 2. S_STFT]\n")

        plt.rcParams['figure.figsize'] = [13, 8]
        plt.rcParams.update({'font.size': 18})

        t, r, z = sig[0][0:(len(sig[0]))], sig[1][0:(len(sig[0]))], sig[2][0:(len(sig[0]))]

        N, s, alpha, fs, n_it, mu = len(r), 100, 4.5, 1, 400, 1e-3

        if input_var == "1":
            tfrx = RSTFRMethod.stft(r, s)  # R component vector #, window=np.gaussian(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2] # 0:sample freq 1: segment times 2:STFT of t
            tfry = RSTFRMethod.stft(t, s)  # T component vector #, window=np.hanning(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2]  # 0:sample freq 1: segment times 2:STFT of r
            tfrz = RSTFRMethod.stft(z, s)  # Z component vector #, window=np.hanning(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2]  # 0:sample freq 1: segment times 2:STFT of z
        else:
            tfrx, tfry, tfrz = RSTFRMethod.stft_s_ist(r, t, z, s, n_it, mu)  # R component vector #, window=np.gaussian(128), nperseg=128, noverlap=91, nfft=128, detrend=False, return_onesided=False)[2] # 0:sample freq 1: segment times 2:STFT of t
            
        nf = np.ceil(N/2).astype(int)
        semi, majo, mino, majon, minon = np.zeros((12,nf,N)), np.zeros((3,nf,N)), np.zeros((3,nf,N)), \
            np.zeros((3,nf,N)), np.zeros((3,nf,N))

        for i in range(N):
            semi[:,:,i], majo[:,:,i], mino[:,:,i], majon[:,:,i], minon[:,:,i] = RSTFRMethod.semimm(np.multiply(10, tfry[:,i]),\
                np.multiply(10, tfrx[:,i]),np.multiply(10, tfrz[:,i]))
            print("Generating the semi major/minor ... ",np.round(i*100/N, 2), "%")


        majornorm, minornorm = np.zeros((nf, N)), np.zeros((nf, N))

        for i in range(nf):
            for j in range(N):
                majornorm[i, j], minornorm[i, j] = np.sqrt(np.dot(majo[:, i, j],majo[:, i, j])), np.sqrt(np.dot(mino[:, i, j],mino[:, i, j]))
            print("Normalising the output values ... ",np.round(i*100/nf, 2), "%")

        
        print(RSTFRMethod.filtering())
        
        cax = np.max((np.max(majornorm), np.max(minornorm)))
        
        plt.imshow(np.abs(majornorm), cmap='hot', alpha=alpha, vmin=0, vmax=0.7*cax)
        plt.title('RS-TFR (SM)')    

        plt.imshow(np.abs(minornorm), cmap="hot", alpha=alpha, vmin=0, vmax=0.7*cax)
        plt.title('RS-TFR (Sm)')

        

        StringIObytes = io.BytesIO()
        plt.savefig(StringIObytes, format='jpg')
        plt.show()

        StringIObytes.seek(0)
        b64jpgdata = base64.b64encode(StringIObytes.read())

        print("Execution time:", time.time()-start_time)

        return np.abs(majornorm), np.abs(minornorm), b64jpgdata