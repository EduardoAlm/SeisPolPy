import numpy as np
import matplotlib.pyplot as plt
import mat4py

data = mat4py.loadmat('examples/dataACRG.mat') # seismic data
data2 =  mat4py.loadmat('examples/datatitanII.ABKT.mat') # seismic data

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

fig.text(0.5, 0.004, 'Time (s)', ha='center')
fig.text(0.004, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.xlabel('')
plt.ylabel('')
plt.sca(axs[0])
plt.plot(data['t'], c='gray', linewidth=1.5, label='t')

plt.sca(axs[1])
plt.plot(data['r'], c='gray', linewidth=1.5, label='r')

plt.sca(axs[2])
plt.plot(data['z'], c='gray', linewidth=1.5, label='z')



plt.tight_layout()
plt.show()
plt.close()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

fig.text(0.5, 0.004, 'Time (s)', ha='center')
fig.text(0.004, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.sca(axs[0])
plt.plot(data2['t'], c='gray', linewidth=1.5, label='t')

plt.sca(axs[1])
plt.plot(data2['r'], c='gray', linewidth=1.5, label='r')

plt.sca(axs[2])
plt.plot(data2['z'], c='gray', linewidth=1.5, label='z')


plt.tight_layout()
plt.show()
plt.close()

data3 = mat4py.loadmat('examples/datagen/vmdata.mat') # seismic data
data4 =  mat4py.loadmat('examples/datagen/acrgdata.mat') # seismic data

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

fig.text(0.5, -0.04, 'Time (s)', ha='center')
fig.text(-0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.sca(axs[0])
plt.plot(data3['t'], c='gray', linewidth=1.5, label='t')

plt.sca(axs[1])
plt.plot(data3['r'], c='gray', linewidth=1.5, label='r')

plt.sca(axs[2])
plt.plot(data3['z'], c='gray', linewidth=1.5, label='z')


plt.tight_layout()
plt.show()
plt.close()

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

fig.text(0.5, 0.04, 'Time (s)', ha='center')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.sca(axs[0])
plt.plot(data4['t'], c='gray', linewidth=1.5, label='t')

plt.sca(axs[1])
plt.plot(data4['r'], c='gray', linewidth=1.5, label='r')

plt.sca(axs[2])
plt.plot(data4['z'], c='gray', linewidth=1.5, label='z')


plt.tight_layout()
plt.show()
plt.close()