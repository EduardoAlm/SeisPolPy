from obspy.clients.syngine import Client
from obspy.geodetics.base import gps2dist_azimuth
import numpy as np
from obspy.signal.rotate import rotate_ne_rt
import scipy.io
from obspy import read
import pyasdf

#ds = pyasdf.ASDFDataSet("examples/datagen/out_titan.h5")
#print(ds)
#print(ds.waveforms.list())
client = Client()
Hypo_lat = -43.01 ; Hypo_lon =  42.17 ; Hypo_depth = 18.5

net = "GE"; sta = "ACRG"; loc = "00"; chan = "BH*"; lat = 5.64; lon = 	-0.21

st = client.get_waveforms(model="ak135f_5s", network=net, station=sta, eventid="GCMT:C201901221901A")
#st = ds.waveforms[sta].synthetics

# st = read('./rssd-2.miniseed')
# st.plot()
# st[3].plot()
# st[4].plot()
#st  = read('./2019-07-06-mw71-central-california-5.miniseed')
baz = gps2dist_azimuth(Hypo_lat, Hypo_lon, lat, lon)
a, b = rotate_ne_rt(st.traces[1].data, st.traces[2].data, np.round(baz[2], 1))
#st.plot()
scipy.io.savemat("examples/datagen/acrgdata.mat", dict(t=b, r=a, z=st.traces[0].data))
print(np.round(baz[2], 1))
st.traces[1].data = a; st.traces[2].data = b;  st.traces[0].data =st.traces[0].data #st_PFO_rem = st_PFO.copy();  # st_FFC_rem.remove_response(output='DISP')
#st.plot()
st.detrend("spline", order=3, dspline=500)

st.decimate(factor=2, strict_length=False)
st.decimate(factor=3, strict_length=False)
starttime = st[0].stats.starttime+60*20
print(starttime)
endtime = st[0].stats.starttime+60*45
print(endtime)
st.trim(starttime=starttime, endtime=endtime)#

###### Synthetic
z = st.traces[0].data
print(len(z))
r = st.traces[1].data
t = st.traces[2].data
name = "examples/data" + sta + ".mat"
scipy.io.savemat(name, dict(t=t, r=r, z=z))
######## Real
# t = st.traces[0].data
# r = st.traces[1].data
# z = st.traces[2].data
# scipy.io.savemat('/home/shazam/SHAZAM/Processing/Codes/High_Res_Pol_Filtering/RSSD.mat', dict(t=t, r=r, z=z))
st.plot()
#