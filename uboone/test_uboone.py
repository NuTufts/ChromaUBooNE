import os,sys
os.environ['PYOPENCL_CTX']='0:1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'
import chroma.api as api
api.use_opencl()
#api.use_cuda()
from unittest_find import unittest
import numpy as np
from chroma.sim import Simulation
from chroma.event import Photons
from chroma.geometry import Surface
import chroma.event as event
from chroma.uboone.uboonedet import ubooneDet
from chroma.gpu.photon import GPUPhotons

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

class TestUbooneDetector(unittest.TestCase):
    def setUp(self):
        # UBOONE
        #daefile = "../gdml/microboone_nowires_chroma_simplified.dae"
        daefile = "dae/microboone_32pmts_nowires_cryostat.dae"
        #daefile = "dae/microboone_32pmts_nowires_cryostat_weldwireplanes.dae"
        self.geo = ubooneDet( daefile, detector_volumes=["vol_PMT_AcrylicPlate","volPaddle_PMT"],
                              wireplane_volumes=[('volTPCPlane_PV0x7f868ac5ef50',add_wireplane_surface)],
                              acrylic_detect=True, acrylic_wls=False,  
                              read_bvh_cache=True, cache_dir="./uboone_cache",
                              dump_node_info=False)
        self.sim = Simulation(self.geo, geant4_processes=0, nthreads_per_block=1, max_blocks=1024)
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
        #nphotons = 7200000
        nphotons = 256*100
        #nphotons = 256*10

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

        if has_root:
            root_file = root_open("output_test_uboone.root", "recreate")
            root_tree = Tree("PhotonData", model=PhotonData )
            root_tree.reset()

        for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False, max_steps=100):
            ev.photons_end.dump_history()
            lht = ev.photons_end[0].last_hit_triangles
            nhits = ev.channels.hit[ np.arange(0,32)[:] ]
            
            #print "nchannels: ",len(ev.channels.hit)
            #print nhits
            #print ev.channels.q
            #print ev.channels.t

            if False:
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
            root_tree.Write()

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
    unittest.main()
    pycuda.driver.stop_profiler()
