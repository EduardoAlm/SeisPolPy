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
data = mat4py.loadmat('examples/dataACRG.mat') # seismic data
sig = np.array([data['t'], data['r'], data['z']])
b64data = SeisPolPy.Flinn.flinn(sig, 50)
f = open("examples/outputFlinnB64.txt", "w")
f.write(b64data)
f.close()
image = base64.b64decode(b64data)
image = io.BytesIO(image)
image = mpimg.imread(image, format='jpg')

plt.imshow(image)
#plt.show()
