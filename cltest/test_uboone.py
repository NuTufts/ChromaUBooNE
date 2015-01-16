import os,sys
os.environ['PYOPENCL_CTX']='1'
from unittest_find import unittest
import numpy as np
import chroma.api as api
from chroma.sim import Simulation
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

        pass

if __name__ == "__main__":
    api.use_opencl()
    unittest.main()
    
