import os,sys,time
os.environ['PYOPENCL_CTX']='0:0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'
import chroma.api as api
api.use_opencl()
#api.use_cuda()

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from chroma.display.pyqtdisplay import PyQtDisplay
from chroma.sim import Simulation
from chroma.event import Photons
import chroma.event

# LOAD CHROMA UBOONE
nthreads_per_block = 1

from uboone import uboone


def gen_photons( nphotons ):

    dphi = np.random.uniform(0,2.0*np.pi, nphotons)
    dcos = np.random.uniform(-1.0, 1.0, nphotons)
    dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), dcos[:] ), dtype=np.float32 )

    pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
    pol = np.zeros_like(pos)
    phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
    pol[:,0] = np.cos(phi)
    pol[:,1] = np.sin(phi)
    pol = np.cross( pol, dir )
    for n,p in enumerate(pol):
        norm = np.sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2] )
        p /= norm

    t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
    wavelengths = np.empty(nphotons, np.float32)
    wavelengths.fill(128.0)

    return Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
    

if __name__ == "__main__":

    app = QtGui.QApplication([])

    start = time.time()
    det = uboone()
    print "[ TIME ] Load detector data ",time.time()-start,"secs"

    display = PyQtDisplay( det )

    print "[ Start Sim. ]"
    start = time.time()
    sim = Simulation(det, geant4_processes=0, nthreads_per_block=nthreads_per_block, max_blocks=1024)
    print "[ TIME ] push geometry data to GPU: ",time.time()-start,"secs"
    nphotons = 256*100
    start = time.time()
    photons = gen_photons( nphotons )
    print "[ TIME ] generate photons ",time.time()-start,"secs"

    start = time.time()
    events = sim.simulate( photons, keep_photons_end=True, max_steps=2000)
    print "[ TIME ] propagate photons ",time.time()-start,"secs"
    
    for ev in events:
        nhits = ev.channels.hit[ np.arange(0,36)[:] ]
        print "Channels with Hits: "
        print nhits
        print "Photoelectrons in each channel: "
        print ev.channels.q
        detected_photons = ev.photons_end.flags[:] & chroma.event.SURFACE_DETECT  # bit-wise AND.  if detected bit set, then value >0, otherwise 0.
        print "Detected photons: ",np.count_nonzero( detected_photons )
        print "hit prep: ",len( ev.photons_end.last_hit_triangles ),len(det.solid_id_to_channel_index),len(det.solid_id)
        channelhit = np.zeros( len(detected_photons), dtype=np.int )
        channelhit[:] = det.solid_id_to_channel_index[ det.solid_id[ ev.photons_end.last_hit_triangles[:] ] ]
        for n,f in enumerate(detected_photons):
            if f!=0:
                # by convention chroma starts event at t=100.0
                print "HIT DETID=",channelhit[n]," POS=",ev.photons_end.pos[n,:]," TIME=",ev.photons_end.t[n]-100.0

        display.plotEvent( ev )


    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

