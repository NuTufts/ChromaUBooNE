import os,sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from chroma.display.pyqtdisplay import PyQtDisplay
from chroma.sim import Simulation
from chroma.event import Photons

# LOAD CHROMA UBOONE
import os,sys
#os.environ['PYOPENCL_CTX']='0:0'
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'
nthreads_per_block = 1

import chroma.api as api
api.use_opencl()                                                                                                                                                                                                     #api.use_cuda()

import numpy as np
from uboone import uboone


if __name__ == "__main__":

    det = uboone()

    app = QtGui.QApplication([])
    display = PyQtDisplay( det )

    print "[ Start Sim. ]"
    sim = Simulation(det, geant4_processes=0, nthreads_per_block=nthreads_per_block, max_blocks=1024)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



# # Generate photons
# nphotons = 256*1000
# dphi = np.random.uniform(0,2.0*np.pi, nphotons)
# dcos = np.random.uniform(-1.0, 1.0, nphotons)
# dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), dcos[:] ), dtype=np.float32 )

# pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
# pol = np.zeros_like(pos)
# phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
# pol[:,0] = np.cos(phi)
# pol[:,1] = np.sin(phi)
# pol = np.cross( pol, dir )
# for n,p in enumerate(pol):
#    norm = np.sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2] )
#    p /= norm

# t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
# wavelengths = np.empty(nphotons, np.float32)
# wavelengths.fill(128.0)

# photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
# print "photons generated"
# events = sim.simulate( photons, keep_photons_end=True, max_steps=2000)
# photon_endpos = None
# photon_colors = None

# colordict = { chroma.event.SURFACE_DETECT:(1.0,0.0,0.0,1.0),
#               chroma.event.BULK_ABSORB:(0.0,1.0,0.0,1.0),
#               chroma.event.WIREPLANE_ABSORB:(0.0,1.0,0.0,0.5),
#               chroma.event.SURFACE_ABSORB:(14.0/255.0,26.0/255.0,191.0/255.0,0.5),
#               0:(0.0,0.0,0.0,0.1)}

# for ev in events:
#     ev.photons_end.dump_history()
#     nphotons = ev.photons_end.pos[:,0:-1].shape[0]
#     flags = ev.photons_end.flags
#     finalstate = flags[:] & chroma.event.SURFACE_DETECT | flags[:] & chroma.event.BULK_ABSORB | flags[:] & chroma.event.WIREPLANE_ABSORB | flags[:] & chroma.event.SURFACE_ABSORB
#     photon_colors = np.zeros( (nphotons,4), dtype=np.float )
#     photon_endpos = ev.photons_end.pos[:,0:-1]
#     weird_flags = []
#     for n,f in enumerate(finalstate):
#         if f==0:
#             if flags[n] not in weird_flags:
#                 weird_flags.append(flags[n])
#         photon_colors[n,0] = colordict[ f ][0]
#         photon_colors[n,1] = colordict[ f ][1]
#         photon_colors[n,2] = colordict[ f ][2]
#         photon_colors[n,3] = colordict[ f ][3]
#     print "weird flags: ",weird_flags
    
# # LOAD PYQTGRAPH
# app = QtGui.QApplication([])
# w = gl.GLViewWidget()
# w.show()
# w.setWindowTitle('pyqtgraph example: Uboone')
# w.setCameraPosition(distance=15000)

# # At world grid
# g = gl.GLGridItem()
# g.scale(2,2,1)
# w.addItem(g)

# detector_meshdata = gl.MeshData( vertexes=geom.mesh.vertices, faces=geom.mesh.triangles )
# detector_meshitem = gl.GLMeshItem( meshdata=detector_meshdata, drawFaces=False, drawEdges=True )
# detector_meshitem.rotate(90, 1, 0, 0 )

# w.addItem( detector_meshitem )

# hits = []
# pos_plot = gl.GLScatterPlotItem( pos=photon_endpos, color=photon_colors, size=2.0, pxMode=True )
# pos_plot.rotate(90,1, 0, 0)
# w.addItem( pos_plot )
# hits.append( pos_plot )

# angle = 2.0
# def update():
#     global angle
#     detector_meshitem.rotate( angle, 0, 0, 1 )
#     for pos in hits:
#         pos.rotate( angle, 0, 0, 1 )
# t = QtCore.QTimer()
# t.timeout.connect(update)
# t.start(50)

# if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#     QtGui.QApplication.instance().exec_()

