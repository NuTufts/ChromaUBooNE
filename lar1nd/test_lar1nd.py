import os,sys
from unittest_find import unittest
import numpy as np
import chroma.api as api
api.use_cuda()
#api.use_opencl()
from chroma.sim import Simulation
from chroma.event import Photons
import chroma.event as event
from chroma.uboone.uboonedet import ubooneDet
try:
    import ROOT as rt
    from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
    from rootpy.io import root_open
    has_root = True
except:
    has_root = False
    raise ValueError("No ROOT")

if has_root:
    class PhotonData( TreeModel ):
        end_x = FloatCol()
        end_y = FloatCol()
        end_z = FloatCol()
        reflect_diffuse  = IntCol()
        reflect_specular = IntCol()
        bulk_scatter     = IntCol()
        bulk_absorb      = IntCol()
        surface_detect   = IntCol()
        surface_absorb   = IntCol()
        surface_reemit   = IntCol()
        
        def reset(self):
            self.reflect_diffuse  = 0
            self.reflect_specular = 0
            self.bulk_scatter     = 0
            self.bulk_absorb      = 0
            self.surface_detect   = 0
            self.surface_absorb   = 0
            self.surface_reemit   = 0

class TestUbooneDetector(unittest.TestCase):
    def setUp(self):
        daefile = "lar1nd_lightguides_nowires_chroma.dae"
        #daefile = "lar1nd_chroma.dae"
        self.geo = ubooneDet( daefile, detector_volumes=["vollightguidedetector"],
                              acrylic_detect=True, acrylic_wls=False,
                              read_bvh_cache=True, cache_dir="./lar1nd_cache")
        self.sim = Simulation(self.geo, geant4_processes=0, nthreads_per_block=256, max_blocks=1000)

    @unittest.skip('skipping testDet')
    def testDet(self):

        # Run only one photon at a time
        nphotons = 100000
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
        nphotons = 50000
        #nphotons = 7200000

        dphi = np.random.uniform(0,2.0*np.pi, nphotons)
        dcos = np.random.uniform(-1.0, 1.0, nphotons)
        dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), dcos[:] ), dtype=np.float32 )

        pos = np.tile([-1000,0,0], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(128.0)
        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
        hit_charges = []

        if has_root:
            root_file = root_open("output_test_lar1nd_wires.root", "recreate")
            root_tree = Tree("PhotonData", model=PhotonData )
            root_tree.reset()
            
        cycles = 0
        for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False, ):
            ev.photons_end.dump_history()
            #lht = ev.photons_end[0].last_hit_triangles
            #nhits = ev.channels.hit[ np.arange(0,30)[:] ]
            if ( cycles%10==0 ):
                print "Cycle: ",cycles
            #print "nchannels: ",len(ev.channels.hit)
            #print nhits
            #print ev.channels.q
            #print ev.channels.t
            cycles += 1
            if True:
                # Fill Tree
                #print "save info for ",len(ev.photons_end)
                for photon in ev.photons_end:
                    root_tree.end_x = photon.pos[0]
                    root_tree.end_y = photon.pos[1]
                    root_tree.end_z = photon.pos[2]

                    root_tree.reflect_diffuse  = int( event.REFLECT_DIFFUSE & photon.flags )
                    root_tree.reflect_specular = int( event.REFLECT_SPECULAR & photon.flags )
                    root_tree.bulk_scatter     = int( event.RAYLEIGH_SCATTER & photon.flags )
                    root_tree.bulk_absorb      = int( event.BULK_ABSORB & photon.flags )
                    root_tree.surface_detect   = int( event.SURFACE_DETECT & photon.flags )
                    root_tree.surface_absorb   = int( event.SURFACE_ABSORB & photon.flags )
                    root_tree.surface_reemit   = int( event.SURFACE_REEMIT & photon.flags )
                    root_tree.fill()
        if has_root:
            root_tree.write()

if __name__ == "__main__":
    unittest.main()
    pass
