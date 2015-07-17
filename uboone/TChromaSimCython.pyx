from cprotobuf import encode_data
import ratchromadata_pb2
from hitPhotons_pb import PhotonHits,Photon
import numpy as np

import chroma.api as api
api.use_cuda()
from chroma.sim import Simulation
from chroma.event import Photons
import chroma.event

from uboone import uboone

cimport numpy as np
import time

import cython
cimport cython

DTYPEINT = np.int
ctypedef np.int_t DTYPEINT_t
DTYPEFLOAT32 = np.float32
ctypedef np.float32_t DTYPEFLOAT32_t
DTYPEUINT16 = np.uint16
ctypedef np.uint16_t DTYPEUINT16_t
DTYPEINT16 = np.int16
ctypedef np.int16_t DTYPEINT16_t
det = uboone()

sim = Simulation(det,geant4_processes=0,nthreads_per_block=128, max_blocks = 1024)

@cython.boundscheck(False)    
cpdef PackPhotonMessage(  np.ndarray[DTYPEFLOAT32_t, ndim=2] pos, 
                        np.ndarray[DTYPEFLOAT32_t, ndim=2] d, 
                        np.ndarray[DTYPEFLOAT32_t, ndim=1] t,
                        np.ndarray[DTYPEFLOAT32_t, ndim=1] wavelength,
                        np.ndarray[DTYPEFLOAT32_t, ndim=2] pol, 
                        np.ndarray[DTYPEINT_t, ndim=1] chanhit,
                        np.ndarray[unsigned int, ndim=1] detect_flag,
                        int nphotons ):
    # we return a dict
    cdef float const1 = float((2*(np.pi)*(6.582*(10**-16))*(299792458.0)))
    cdef float const2 = (4.135667516 * (10**-21))

    phits = PhotonHits()

    cdef int nhits = 0         
    for x in range(len(detect_flag)):
        if detect_flag[x]!=0:
            phits.photon.add( PMTID=chanhit[x],
                              Time= t[x],
                              KineticEnergy =  const1/wavelength[x],
                              posX = pos[x,0],
                              posY = pos[x,1],
                              posZ = pos[x,2],
                              momX = const2/wavelength[x]*d[x,0],
                              momY = const2/wavelength[x]*d[x,1],
                              momZ = const2/wavelength[x]*d[x,2],
                              polX = pol[x,0],
                              polY = pol[x,1],
                              polZ = pol[x,2],
                              trackID = x,
                              origin = Photon.CHROMA )
            nhits += 1
    print "Packed ",nhits,"hits"
    phits.count = nhits
    return phits.SerializeToString()
                      
    
@cython.boundscheck(False)    
cpdef MakePhotonMessage(chromaData):
    # in this version, we use cprotobuf
    photons = GenScintPhotons(chromaData)
    events = sim.simulate(photons, keep_photons_end=True, max_steps=2000)
    stime = time.clock()
    cdef float const1 = float((2*(np.pi)*(6.582*(10**-16))*(299792458.0)))
    cdef float const2 = (4.135667516 * (10**-21))
    cdef int n, f
    cdef np.ndarray[DTYPEINT_t,ndim = 1] channelhit	
    cdef np.ndarray[unsigned int,ndim = 1] detected_photons
    cdef bytearray msg = bytearray()

    phits = None
    for ev in events:
        detected_photons = ev.photons_end.flags[:] & <unsigned int>(chroma.event.SURFACE_DETECT)
        channelhit = np.zeros(len(detected_photons),dtype = DTYPEINT)
        channelhit[:] = det.solid_id_to_channel_index[ det.solid_id[ev.photons_end.last_hit_triangles[:] ] ]
        nhits = int(np.count_nonzero(detected_photons))
        print "Number of hits: ",nhits
        ptime = time.time()
        
        phits = PackPhotonMessage( ev.photons_end.pos,
                                   ev.photons_end.dir,
                                   ev.photons_end.t,
                                   ev.photons_end.wavelengths,
                                   ev.photons_end.pol,
                                   channelhit,
                                   detected_photons,
                                   len(detected_photons) )
        print "TIME TO PACK: ",time.time()-ptime
                                   
    etime = time.clock()
    print "TIME TO MAKE MESSAGE: ",(etime-stime)
    return str(phits)

@cython.boundscheck(False)
def GenScintPhotons(protoString):
    stime = time.clock()
    cdef int nsphotons,stepPhotons,i,j
    cdef np.ndarray[DTYPEFLOAT32_t,ndim = 2] pos, pol
    cdef np.ndarray[DTYPEFLOAT32_t,ndim = 1] wavelengths, t
    nsphotons = 0
    for i,sData in enumerate(protoString.stepdata):
        nsphotons += sData.nphotons
    print "NSPHOTONS: ",nsphotons
    pos = np.zeros((nsphotons,3),dtype=np.float32)
    pol = np.zeros_like(pos)
    t = np.zeros(nsphotons, dtype=np.float32)
    wavelengths = np.empty(nsphotons, np.float32)
    wavelengths.fill(128.0)
    dphi = np.random.uniform(0,2.0*np.pi, nsphotons)
    dcos = np.random.uniform(-1.0, 1.0, nsphotons)
    phi = np.random.uniform(0,2.0*np.pi, nsphotons).astype(np.float32)
    dir = np.array( zip( np.sqrt(1-dcos[:]*dcos[:])*np.cos(dphi[:]), np.sqrt(1-dcos[:]*dcos[:])*np.sin(dphi[:]), dcos[:] ), dtype=np.float32 )
    pol[:,0] = np.cos(phi)
    pol[:,1] = np.sin(phi)
    stepPhotons = 0
    for i,sData in enumerate(protoString.stepdata):
        #instead of appending to array every loop, the full size (nsphotons x 3) is allocated to begin, then 
        #values are filled properly by incrementing stepPhotons.
        for j in xrange(stepPhotons, (stepPhotons+sData.nphotons)):
            pos[j,0] = np.random.uniform(sData.step_start_x,sData.step_end_x)
            pos[j,1] = np.random.uniform(sData.step_start_y,sData.step_end_y)
            pos[j,2] = np.random.uniform(sData.step_start_z,sData.step_end_z)
        #moved pol outside of the loop. saved ~3 seconds in event loop time. 
        # for j in xrange(stepPhotons, (stepPhotons+sData.nphotons)):
        #     pol[j,0] = np.random.uniform(0,((1/3.0)**.5))
        #     pol[j,1] = np.random.uniform(0,((1/3.0)**.5))
        #     pol[j,2] = ((1 - pol[j,0]**2 - pol[j,1]**2)**.5)
        for j in xrange(stepPhotons, (stepPhotons+sData.nphotons)):
            t[j] = (np.random.exponential(1/45.0) + (sData.step_end_t-sData.step_start_t))
        stepPhotons += sData.nphotons
    etime = time.clock()
    print "TIME TO GEN PHOTONS: ",(etime-stime)
    return Photons(pos = pos, pol = pol, t = t, dir = dir, wavelengths = wavelengths)




