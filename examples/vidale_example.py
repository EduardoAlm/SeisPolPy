import SeisPolPy
import mat4py
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import matplotlib.image as mpimg
import pyasdf

#ds = pyasdf.ASDFDataSet("examples/out_vm.h5")
#print(ds)
#print(ds.waveforms.list())
#data = ds.waveforms['II.ABKT'].synthetics
#sig = np.array([data.traces[0].data, data.traces[1].data, data.traces[2].data])
data = mat4py.loadmat('examples/dataII.ABKTvm.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
b64data1, b64data2 = SeisPolPy.Vidale.vidale(sig, 50)

f = open("examples/outputVidaleB641.txt", "w")
f.write(b64data1)
f.close()
f1 = open("examples/outputVidaleB642.txt", "w")
f1.write(b64data2)
f1.close()


image1 = base64.b64decode(b64data1)
image1 = io.BytesIO(image1)
image1 = mpimg.imread(image1, format='jpg')
plt.imshow(image1, interpolation='nearest')
plt.show()
plt.close()

image2 = base64.b64decode(b64data2)
image2 = io.BytesIO(image2)
image2 = mpimg.imread(image2, format='jpg')
plt.imshow(image2, interpolation='nearest')
plt.show()
plt.close()