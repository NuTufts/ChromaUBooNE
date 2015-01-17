import os,sys
from unittest_find import unittest
import numpy as np
import chroma.api as api
api.use_cuda()
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
        self.geo = ubooneDet( "../gdml/microboone_nowires_chroma_simplified.dae",  acrylic_detect=True, acrylic_wls=False )
        self.sim = Simulation(self.geo, geant4_processes=0)

    def testDet(self):

        # Run only one photon at a time
        nphotons = 1
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
            ev.photons_end.dump()
            lht = ev.photons_end[0].last_hit_triangles
            print ev.photons_end[0].last_hit_triangles
            print self.geo.material1_index[lht], self.geo.material2_index[lht]

if __name__ == "__main__":

    unittest.main()
    
