import os,sys
os.environ['PYOPENCL_CTX']='1'
from unittest_find import unittest
import numpy as np

import chroma.api as api
import chroma.gpu.tools as gputools
from chroma.geometry import Solid, Geometry, vacuum
from chroma.loader import create_geometry_from_obj
from chroma.detector import Detector
from chroma.make import box
from chroma.sim import Simulation
from chroma.event import Photons

from chroma.demo.optics import r7081hqe_photocathode
try:
    import ROOT as rt
    has_root = True
except:
    has_root = False
class TestDetector(unittest.TestCase):
    def setUp(self):
        # Setup geometry
        cube = Detector(vacuum)
        cube.add_pmt(Solid(box(10.0,10.0,10.0), vacuum, vacuum, surface=r7081hqe_photocathode))
        cube.set_time_dist_gaussian(1.2, -6.0, 6.0)
        cube.set_charge_dist_gaussian(1.0, 0.1, 0.5, 1.5)

        geo = create_geometry_from_obj(cube, update_bvh_cache=True, read_bvh_cache=False)
        print "Number of channels in detector: ",geo.num_channels()
        self.geo = geo
        self.sim = Simulation(self.geo, geant4_processes=0)

        self.rfile = rt.TFile("output_test_detector.root","recreate")
        self.htime = rt.TH1D("htime","Time;ns",120, 80, 120 )
        self.hcharge = rt.TH1D("hcharge","Charge;pe",100, 0.5, 1.5 )

    @unittest.skip('Skipping time test')
    def testTime(self):
        '''Test PMT time distribution'''

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
        wavelengths.fill(400.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t,
                          wavelengths=wavelengths)

        hit_times = []
        for ev in self.sim.simulate( (photons for i in xrange(500)), keep_photons_end=True, keep_photons_beg=False):
            for n,hit in enumerate(ev.channels.hit):
                if hit:
                    hit_times.append(ev.channels.t[n])
                    print "Hits from photon %d: "%(n),ev.photons_end.pos[n]," with t=",ev.channels.t[n]," q=",ev.channels.q[n],". starting from ",ev.photons_beg.pos[n], " Hit=",ev.channels.hit[n]
                    self.htime.Fill( ev.channels.t[n] )
                    self.hcharge.Fill( ev.channels.q[n] )
        hit_times = np.array(hit_times)

        self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)

    #@unittest.skip('Ray data file needs to be updated')
    def testCharge(self):
        '''Test PMT charge distribution'''

        # Run only one photon at a time
        phi= 1.0
        theta = 1.0
        nphotons = 1
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([np.sin(theta*np.pi/180.0)*np.cos(phi*np.pi/180.0), np.sin(theta*np.pi/180.0)*np.sin(phi*np.pi/180.0), np.cos(theta*np.pi/180.0)], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32)+100.0
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(400.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)

        hit_charges = []
        for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False):
            if ev.channels.hit[0]:
                hit_charges.append(ev.channels.q[0])
                self.hcharge.Fill( ev.channels.q[0] )
                self.htime.Fill( ev.channels.t[0] )
                print "Hits:  with q=",ev.channels.q[0],". Hit=",ev.channels.hit[0]
            #ev.photons_end.dump()
        hit_charges = np.array(hit_charges)
        
        self.assertAlmostEqual(hit_charges.mean(),  1.0, delta=1e-1)
        self.assertAlmostEqual(hit_charges.std(), 0.1, delta=1e-1)
    def tearDown(self):
        self.rfile.Write()

if __name__ == "__main__":
    api.use_opencl()
    unittest.main()

