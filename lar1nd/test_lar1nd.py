import os,sys,time
from unittest_find import unittest
import array
import numpy as np
import chroma.api as api
api.use_cuda()
#api.use_opencl()
from chroma.sim import Simulation
from chroma.event import Photons
import chroma.event as event
from chroma.geometry import Surface
from chroma.uboone.uboonedet import ubooneDet
from chroma.uboone.daq_uboone import GPUDaqUBooNE
try:
    import ROOT as rt
    from rootpy.tree import Tree, TreeModel, FloatCol, IntCol, FloatArrayCol
    from rootpy.io import root_open
    has_root = True
except:
    has_root = False
    raise ValueError("No ROOT")

# CHANNEL INFORMATION
GPUDaqUBooNE.NTDC = 10000
GPUDaqUBooNE.NS_PER_TDC = 1.0
NCHANNELS = 1000

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
    class OpDet( TreeModel ):
        eventid = IntCol()
        id = IntCol()
        NTDC = IntCol()
        NS_PER_TDC = FloatCol()
        adc = FloatArrayCol(GPUDaqUBooNE.NTDC)
        q = FloatCol()
        t = FloatCol()
    class OpMap( TreeModel ):
        opid = IntCol()
        x = FloatCol()
        y = FloatCol()
        z = FloatCol()

lar1nd_wireplane = Surface( 'lar1nd_wireplane' )
lar1nd_wireplane.nplanes = 3.0
lar1nd_wireplane.wire_pitch = 0.3
lar1nd_wireplane.wire_diameter = 0.015
lar1nd_wireplane.transmissive = 1
lar1nd_wireplane.model = Surface.SURFACE_WIREPLANE

def add_wireplane_surface( solid ):
    # function detector class will use to add a wireplane surface to the geometry
    # LAr1ND has two drift regions, so we need two planes
    # set surface for triangles on x=-2023.25 and x=2023.25 planes
    
    for n,triangle in enumerate(solid.mesh.triangles):
        #print [ solid.mesh.vertices[x] for x in triangle ] # for debug
        nxplane = 0
        for ivert in triangle:
            if solid.mesh.vertices[ivert,0]==-2023.25 or solid.mesh.vertices[ivert,0]==2023.25:
                nxplane += 1
        # if the numbr of vertices have the correct x value, we say we have the right traingle
        if nxplane==3: 
            print [ solid.mesh.vertices[x] for x in triangle ]
            solid.surface[ n ] = lar1nd_wireplane
            solid.unique_surfaces = np.unique( solid.surface )

class TestUbooneDetector(unittest.TestCase):
    def setUp(self):
        daefile = "dae/lar1nd_lightguides_nowires_chroma.dae" # without wires
        #daefile = "dae/lar1nd_chroma.dae" # with wires
        self.geo = ubooneDet( daefile, detector_volumes=["vollightguidedetector"],
                              wireplane_volumes=[('volTPCPlaneVert_PV0x7fdcd2728c70',add_wireplane_surface)],
                              acrylic_detect=True, acrylic_wls=False,
                              read_bvh_cache=True, cache_dir="./lar1nd_cache",
                              dump_node_info=False )
        self.sim = Simulation(self.geo, geant4_processes=0, nthreads_per_block=192, max_blocks=1024, user_daq=GPUDaqUBooNE )

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
        nphotons_test = 256*1000
        #nphotons = 7200000


        if has_root:
            root_file = root_open("output_test_lar1nd_scanx_high_stats.root", "recreate")
            root_tree = Tree("PhotonData", model=PhotonData )
            root_tree.reset()
            root_channels = Tree("OpDet", model=OpDet )
            root_opmap = Tree("OpMap", model=OpMap)
            channelmap = self.sim.gpu_geometry.solid_id_to_channel_index_gpu.get()
            channels = np.argwhere(channelmap>-1)
            nchannels = NCHANNELS
            channeldict = dict( zip( range(0,nchannels), channels.ravel().tolist() ) )
            for ich in range(0,nchannels):
                root_opmap.opid = ich
                solid = self.sim.detector.solids[channeldict[ich]]
                root_opmap.x = np.sum( solid.mesh.vertices[:,0] )/len( solid.mesh.vertices )
                root_opmap.y = np.sum( solid.mesh.vertices[:,1] )/len( solid.mesh.vertices )
                root_opmap.z = np.sum( solid.mesh.vertices[:,2] )/len( solid.mesh.vertices )
                root_opmap.fill()
            root_opmap.write()
            

        for eventid in xrange(0,102):
            print "Event: ",eventid

            if eventid<101:
                nphotons = nphotons_test*20
                z = -200 + 4*eventid
            else:
                # reference
                nphotons = nphotons_test
                z = 0 

            t_photon_start = time.time()
            dphi = np.random.uniform(0,2.0*np.pi, nphotons)
            dcos = np.random.uniform(-1.0, 1.0, nphotons)
            dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), dcos[:] ), dtype=np.float32 )
            pos = np.tile([-1000+z,0,0], (nphotons,1)).astype(np.float32)
            pol = np.zeros_like(pos)
            phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
            pol[:,0] = np.cos(phi)
            pol[:,1] = np.sin(phi)
            t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
            wavelengths = np.empty(nphotons, np.float32)
            wavelengths.fill(128.0)
            photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
            t_end_start = time.time()
            print "define photon time: ",t_end_start-t_photon_start,"sec"
            
            for ev in self.sim.simulate( (photons for i in xrange(1)), keep_photons_end=True, keep_photons_beg=False, ):
                #ev.photons_end.dump_history()
                #print ev.channels.t
                if ( eventid%10==0 ):
                    print "Event: ",eventid
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
                if True:
                    root_channels.eventid = eventid
                    t_root_start = time.time()
                    for ichannel in xrange(0,NCHANNELS):
                        root_channels.id = ichannel
                        root_channels.NTDC = GPUDaqUBooNE.NTDC
                        root_channels.NS_PER_TDC = GPUDaqUBooNE.NS_PER_TDC
                        channeladc =  ev.channels.q[ GPUDaqUBooNE.NTDC*ichannel: (ichannel+1)*GPUDaqUBooNE.NTDC]
                        root_channels.adc[:] = array.array( 'f', channeladc[:].ravel().tolist() )[:]
                        root_channels.q = np.sum( channeladc )
                        #if root_channels.q>0:
                        #    print channeladc[ np.where( channeladc>0.0 ) ]
                        root_channels.t = ev.channels.t[ichannel]
                        root_channels.fill()
                    t_root_end = time.time()
                    print "ROOT Fill time: ",t_root_end-t_root_start," sec"
        if has_root:
            root_tree.write()
            root_channels.write()

if __name__ == "__main__":
    unittest.main()
    pass
