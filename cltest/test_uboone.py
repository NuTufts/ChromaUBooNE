import os,sys
#os.environ['PYOPENCL_CTX']='1'
from unittest_find import unittest
import numpy as np
import chroma.api as api
api.use_opencl()
from chroma.sim import Simulation
from chroma.event import Photons
from chroma.uboone.uboonedet import ubooneDet
try:
    import ROOT as rt
    has_root = True
except:
    has_root = False

class TestUbooneDetector(unittest.TestCase):
    def setUp(self):
        #self.geo = ubooneDet( "../gdml/microboone_nowires_chroma_simplified.dae",  acrylic_detect=False, acrylic_wls=True )
        self.geo = ubooneDet( "../gdml/microboone_nowires_chroma_simplified.dae",  acrylic_detect=True, acrylic_wls=False,  read_bvh_cache=False )
        self.sim = Simulation(self.geo, geant4_processes=0)
        self.origin = self.geo.bvh.world_coords.world_origin

        print "Triangles"
        print self.geo.mesh.triangles

        print "Vertices"
        print self.geo.mesh.vertices

        print "number of surfaces: ",len( self.geo.surface_index )
        print "number unique surfces: ", len( self.geo.unique_surfaces )
        for i in xrange(0,5):
            print "surface %d: "%(i), np.sum( np.where( self.geo.surface_index==i, 1, 0 ) )

    @unittest.skip('skipping testDet')
    def testDet(self):

        # Run only one photon at a time
        nphotons = 1000000
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(128.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
        hit_charges = []
        for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False, ):
            ev.photons_end.dump_history()
            lht = ev.photons_end[0].last_hit_triangles
            print "LHT: ",lht

    def testPhotonBomb(self):

        # Run only one photon at a time
        nphotons = 1000000

        dphi = np.random.uniform(0,2.0*np.pi, nphotons)
        dcos = np.random.uniform(-1.0, 1.0, nphotons)
        dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), dcos[:] ), dtype=np.float32 )

        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(128.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
        hit_charges = []
        for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False, ):
            ev.photons_end.dump_history()
            lht = ev.photons_end[0].last_hit_triangles
            nhits = ev.channels.hit[ np.arange(0,30)[:] ]
            
            print "nchannels: ",len(ev.channels.hit)
            print nhits
            print ev.channels.q
            print ev.channels.t

if __name__ == "__main__":
    unittest.main()
    
