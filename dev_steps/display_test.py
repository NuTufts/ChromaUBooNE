import os,sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# LOAD PYQTGRAPH
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: Uboone')
w.setCameraPosition(distance=15000)

# At world grid
g = gl.GLGridItem()
g.scale(2,2,1)
w.addItem(g)

# LOAD UBOONE


import os,sys
os.environ['PYOPENCL_CTX']='0:1'
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'

import chroma.api as api
api.use_opencl()
#api.use_cuda()

import numpy as np
from chroma.sim import Simulation
from chroma.event import Photons
import chroma.event
from chroma.geometry import Surface
from chroma.uboone.uboonedet import ubooneDet
from photon_fromstep import GPUPhotonFromSteps

uboone_wireplane = Surface( 'uboone_wireplane' )
uboone_wireplane.nplanes = 3.0
uboone_wireplane.wire_pitch = 0.3
uboone_wireplane.wire_diameter = 0.015
uboone_wireplane.transmissive = 1
uboone_wireplane.model = Surface.SURFACE_WIREPLANE


def add_wireplane_surface( solid ):
    # function detector class will use to add a wireplane surface to the geometry
    # set surface for triangles on x=-1281.0 plane
    for n,triangle in enumerate(solid.mesh.triangles):
        nxplane = 0
        for ivert in triangle:
            if solid.mesh.vertices[ivert,0]==-1281.0:
                nxplane += 1
        if nxplane==3:
            print [ solid.mesh.vertices[x] for x in triangle ]
            solid.surface[ n ] = uboone_wireplane
            solid.unique_surfaces = np.unique( solid.surface )

daefile = "./../uboone/dae/microboone_32pmts_nowires_cryostat.dae"
geom = ubooneDet( daefile, detector_volumes=["vol_PMT_AcrylicPlate","volPaddle_PMT"],
                  #wireplane_volumes=[('volTPCPlane_PV0x7f868ac5ef50',add_wireplane_surface)],
                  acrylic_detect=True, acrylic_wls=False,
                  read_bvh_cache=True, cache_dir="./../uboone/uboone_cache",
                  dump_node_info=False)
sim = Simulation(geom, geant4_processes=0, nthreads_per_block=1, max_blocks=1024)
geom.mesh.vertices
geom.mesh.triangles

detector_meshdata = gl.MeshData( vertexes=geom.mesh.vertices, faces=geom.mesh.triangles )
detector_meshitem = gl.GLMeshItem( meshdata=detector_meshdata, drawFaces=False, drawEdges=True )
detector_meshitem.rotate(90, 1, 0, 0 )

w.addItem( detector_meshitem )

steps = np.load( 'steps.npy' )

# draw steps
step_plot = gl.GLLinePlotItem( pos=steps[:,0:3], color=(1.0,1.0,1.0,1.0), width=2.0, mode='line_strip' )
step_plot.rotate(90, 1, 0, 0)
w.addItem( step_plot )

gpuphotons = GPUPhotonFromSteps( steps, cl_context=sim.context )
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
photons = gpuphotons.get()
print "photons generated"

events = sim.simulate( photons, keep_photons_end=True, max_steps=2000)
positions = []

colordict = { chroma.event.SURFACE_DETECT:(1.0,0.0,0.0,1.0),
              chroma.event.BULK_ABSORB:(0.0,1.0,0.0,1.0),
              chroma.event.WIREPLANE_ABSORB:(0.0,1.0,0.0,0.5),
              chroma.event.SURFACE_ABSORB:(14.0/255.0,26.0/255.0,191.0/255.0,0.5),
              0:(0.0,0.0,0.0,0.1)}

for ev in events:
    ev.photons_end.dump_history()
    nphotons = ev.photons_end.pos[:,0:-1].shape[0]
    flags = ev.photons_end.flags
    finalstate = flags[:] & chroma.event.SURFACE_DETECT | flags[:] & chroma.event.BULK_ABSORB | flags[:] & chroma.event.WIREPLANE_ABSORB | flags[:] & chroma.event.SURFACE_ABSORB
    photon_colors = np.zeros( (nphotons,4), dtype=np.float )
    weird_flags = []
    for n,f in enumerate(finalstate):
        if f==0:
            if flags[n] not in weird_flags:
                weird_flags.append(flags[n])
        photon_colors[n,0] = colordict[ f ][0]
        photon_colors[n,1] = colordict[ f ][1]
        photon_colors[n,2] = colordict[ f ][2]
        photon_colors[n,3] = colordict[ f ][3]
    print "weird flags: ",weird_flags
    
    pos_plot = gl.GLScatterPlotItem( pos=ev.photons_end.pos[:,0:-1], color=photon_colors, size=2.0, pxMode=True )
    pos_plot.rotate(90,1, 0, 0)
    positions.append( pos_plot )
    w.addItem( pos_plot )

angle = 2.0
def update():
    global angle
    detector_meshitem.rotate( angle, 0, 0, 1 )
    step_plot.rotate( angle, 0, 0, 1 )
    for pos in positions:
        pos.rotate( angle, 0, 0, 1 )
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()

