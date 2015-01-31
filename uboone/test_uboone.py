import os,sys
#os.environ['PYOPENCL_CTX']='0:1'
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'
import chroma.api as api
api.use_opencl()
#api.use_cuda()
from unittest_find import unittest
import numpy as np
from chroma.sim import Simulation
from chroma.event import Photons
from chroma.uboone.uboonedet import ubooneDet
from chroma.gpu.photon import GPUPhotons

try:
    import ROOT as rt
    has_root = True
except:
    has_root = False

class TestUbooneDetector(unittest.TestCase):
    def setUp(self):
        # UBOONE
        #daefile = "../gdml/microboone_nowires_chroma_simplified.dae"
        daefile = "dae/microboone_32pmts_nowires_cryostat.dae"
        self.geo = ubooneDet( daefile, detector_volumes=["vol_PMT_AcrylicPlate","volPaddle_PMT"],
                              acrylic_detect=True, acrylic_wls=False,  
                              read_bvh_cache=True, cache_dir="./uboone_cache",
                              dump_node_info=True)
        self.sim = Simulation(self.geo, geant4_processes=0)
        self.origin = self.geo.bvh.world_coords.world_origin


    @unittest.skip('skipping testDet')
    def testDet(self):

        # Run only one photon at a time
        nphotons = 1000000
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol = np.cross( (np.cos(phi), np.sin(phi),0), dir )

        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        pol = np.cross( pol, dir )
        for n,p in enumerate(pol):
            norm = np.sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2] )
            p /= norm
        #print pol
        t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(128.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
        hit_charges = []
        for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False, ):
            ev.photons_end.dump_history()
            lht = ev.photons_end[0].last_hit_triangles

    #@unittest.skip('skipping testDet')
    def testPhotonBomb(self):

        # Run only one photon at a time
        nphotons = 256*1000

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

    @unittest.skip('skipping testDet')
    def testWorkQueue(self):

        # Run only one photon at a time
        nphotons = 32*12

        dphi = np.random.uniform(0,2.0*np.pi, nphotons)
        dcos = np.random.uniform(-1.0, 1.0, nphotons)
        dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), 
                             np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), 
                             dcos[:] ), dtype=np.float32 )
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

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)

        #rq = RayQueue( self.sim.context )
        #rq.checknodes.print_dev_info()
        #rq.simulate( photons, self.sim )

if __name__ == "__main__":
    import pycuda
    unittest.main()
    pycuda.driver.stop_profiler()
