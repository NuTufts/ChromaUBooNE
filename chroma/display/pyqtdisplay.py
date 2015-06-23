import os,sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import chroma.event
from chroma.event import Event
from chroma.geometry import Geometry
import numpy as np


class PyQtDisplay():

    photon_colordict = { chroma.event.SURFACE_DETECT:(1.0,0.0,0.0,1.0),
                         chroma.event.BULK_ABSORB:(0.0,1.0,0.0,1.0),
                         chroma.event.WIREPLANE_ABSORB:(0.0,1.0,0.0,0.5),
                         chroma.event.SURFACE_ABSORB:(14.0/255.0,26.0/255.0,191.0/255.0,0.5),
                         0:(0.0,0.0,0.0,0.1)}

    def __init__( self, detector ):
        geom = detector
        self.detector_meshdata = gl.MeshData( vertexes=geom.mesh.vertices, faces=geom.mesh.triangles )
        self.detector_meshitem = gl.GLMeshItem( meshdata=self.detector_meshdata, drawFaces=False, drawEdges=True )
        self.detector_meshitem.rotate(90, 1, 0, 0 )

        # GL Window
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('PyQt Geometry Display')
        self.window.setCameraPosition(distance=15000)
        self.show()
        
        # 3D Mesh Scene
        self.window.addItem( self.detector_meshitem )
        
        # Dummy photon hits
        nphotons = 10
        photon_end_pos = np.zeros( (nphotons,3), dtype=np.float )
        photon_colors = np.zeros( (nphotons,4), dtype=np.float )
        self.hit_plot = gl.GLScatterPlotItem( pos=photon_end_pos, color=photon_colors, size=2.0, pxMode=True )
        self.hit_plot.rotate(90,1, 0, 0)
        self.window.addItem( self.hit_plot )

    def show(self):
        self.window.show()
        
    def plotEvent( self, ev ):
        ev.photons_end.dump_history()
        nphotons = ev.photons_end.pos[:,0:-1].shape[0]
        print "plot ",nphotons," photons"
        flags = ev.photons_end.flags
        finalstate = flags[:] & chroma.event.SURFACE_DETECT | flags[:] & chroma.event.BULK_ABSORB | flags[:] & chroma.event.WIREPLANE_ABSORB | flags[:] & chroma.event.SURFACE_ABSORB
        photon_colors = np.zeros( (nphotons,4), dtype=np.float )
        photon_endpos = ev.photons_end.pos[:,0:-1]
        weird_flags = []
        for n,f in enumerate(finalstate):
            if f==0:
                if flags[n] not in weird_flags:
                    weird_flags.append(flags[n])
            photon_colors[n,0] = PyQtDisplay.photon_colordict[ f ][0]
            photon_colors[n,1] = PyQtDisplay.photon_colordict[ f ][1]
            photon_colors[n,2] = PyQtDisplay.photon_colordict[ f ][2]
            photon_colors[n,3] = PyQtDisplay.photon_colordict[ f ][3]

        self.hit_plot.setData( pos=photon_endpos, color=photon_colors )

    def plotEvents(self,events):
        for ev in events:
            self.plotEvent(ev)
            print "Hit [ENTER] to plot next event."
            raw_input()

#if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#    QtGui.QApplication.instance().exec_()

