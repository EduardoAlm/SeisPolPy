import SeisPolPy
import mat4py
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import matplotlib.image as mpimg

data = mat4py.loadmat('examples/ACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
b64dataM, b64datam = SeisPolPy.Rstfr.rstfr(sig, "s_stft")

f = open("examples/outputRstfrB64_S_STFTMajor.txt", "w")
f.write(b64dataM)
f.close()
f1 = open("examples/outputRstfrB64_S_STFTMinor.txt", "w")
f1.write(b64datam)
f1.close()

imageM = base64.b64decode(b64dataM)
imageM = io.BytesIO(imageM)
imageM = mpimg.imread(imageM, format='jpg')
plt.imshow(imageM, interpolation='nearest')
plt.show()
plt.close()

imagem = base64.b64decode(b64datam)
imagem = io.BytesIO(imagem)
imagem = mpimg.imread(imagem, format='jpg')
plt.imshow(imagem, interpolation='nearest')
plt.show()