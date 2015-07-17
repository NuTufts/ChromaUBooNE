import os,sys,time,random
import numpy as np
import chroma.api as api
if api.is_gpu_api_cuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray as ga
elif api.is_gpu_api_opencl():
    import pyopencl as cl
    import pyopencl.array as ga
from chroma.tools import profile_if_possible
import chroma.event
from chroma.gpu.tools import get_module, api_options, chunk_iterator, to_float3, copy_to_float3, get_rng_states
from chroma.gpu.gpufuncs import GPUFuncs
from chroma.gpu.photon import GPUPhotons

class GPUPhotonFromSteps( GPUPhotons ):
    """
    This class loads its gpu arrays of photons from a numpy array holding 
    information on particle steps taken in a detector
    for each step, the number of photons to be produced are stored.
    We take the steps and produce the photons, which can be propagated later
    using GPUPhotons::propagate
    """
    _x1 = 0
    _y1 = 1
    _z1 = 2
    _x2 = 3
    _y2 = 4
    _z2 = 5
    _nphotons = 6
    _fsratio = 7
    _fconst = 8
    _sconst = 9
    def __init__(self, steps_arr, multiple=1.0, nthreads_per_block=64, max_blocks=1024, ncopies=1,
                 seed=None, cl_context=None):
        """
        Generates photons from information in the steps_arr
        
        Parameters
        ----------
        steps_arr : numpy.array with shape=(N,10) dtype=np.float
           contains [ x1, y1, z1, x2, y2, z2, nphotons, fast_to_slow_ratio, fast_time_constatn, slow_time_constatn ]
           in the future could generalize this to many different time components.
           developed for liquid argon TPCs.
        multiple : float
           scale up the number of photons generated (not implemented yet)
        """
        self.steps_array = steps_arr
        self.nsteps = self.steps_array.shape[0]
        if multiple!=1.0:
            raise RuntimeError('Have not implemented scaling of the number of photons generated.')

        # ===========================
        # GEN PHOTONS
        tstart_genphotons =  time.time()
        # we do the dumbest thing first (i.e., no attempt to do fancy GPU manipulations here)
        # on the CPU, we scan the steps to determine the total number of photons using poisson statistics
        # we assume the user has seeded the random number generator to her liking
        tstart_nphotons = time.time()
        self.step_fsratio = np.array( self.steps_array[:,self._fsratio], dtype=np.float32 )
        self.nphotons_per_step = np.array( [ np.random.poisson( z ) for z in self.steps_array[:,self._nphotons].ravel() ], dtype=np.int )
        self.nphotons = reduce( lambda x, y : x + y, self.nphotons_per_step.ravel() )
        print "NSTEPS: ",self.nsteps
        print "NPHOTONS: ",self.nphotons," (time to determine per step=",time.time()-tstart_nphotons
        # now we make an index array for which step we need to get info from
        self.source_step_index = np.zeros( self.nphotons, dtype=np.int32 )
        current_index=0
        for n, n_per_step in enumerate( self.nphotons_per_step ):
            self.source_step_index[current_index:current_index+n_per_step] = n
            current_index += n_per_step
        # push everything to the GPU
        tstart_transfer = time.time()
        if api.is_gpu_api_cuda():
            # step info
            self.step_pos1_gpu = ga.empty(shape=self.nsteps, dtype=ga.vec.float3)
            self.step_pos2_gpu = ga.empty(shape=self.nsteps, dtype=ga.vec.float3)
            self.step_fsratio_gpu = ga.to_gpu( self.step_fsratio )
            self.source_step_index_gpu = ga.to_gpu( self.source_step_index )
            # photon info
            self.pos = ga.empty( shape=self.nphotons, dtype=ga.vec.float3 )
            self.dir = ga.empty( shape=self.nphotons, dtype=ga.vec.float3 )
            self.pol = ga.empty( shape=self.nphotons, dtype=ga.vec.float3 )
            self.wavelengths = ga.empty(shape=self.nphotons*ncopies, dtype=np.float32)
            self.t = ga.to_gpu( 100.0*np.ones(self.nphotons*ncopies, dtype=np.float32) )
            self.last_hit_triangles = ga.empty(shape=self.nphotons*ncopies, dtype=np.int32)
            self.flags = ga.empty(shape=self.nphotons*ncopies, dtype=np.uint32)
            self.weights = ga.empty(shape=self.nphotons*ncopies, dtype=np.float32)
        elif api.is_gpu_api_opencl():
            cl_queue = cl.CommandQueue( cl_context )
            # step info
            self.step_pos1_gpu = ga.empty(cl_queue, self.nsteps, dtype=ga.vec.float3)
            self.step_pos2_gpu = ga.empty(cl_queue, self.nsteps, dtype=ga.vec.float3)
            self.step_fsratio_gpu  = ga.to_device( cl_queue, self.step_fsratio )
            self.source_step_index_gpu = ga.to_device( cl_queue, self.source_step_index )
            # photon info
            self.pos = ga.empty( cl_queue, self.nphotons, dtype=ga.vec.float3 )
            self.dir = ga.empty( cl_queue, self.nphotons, dtype=ga.vec.float3 )
            self.pol = ga.empty( cl_queue, self.nphotons, dtype=ga.vec.float3 )
            self.wavelengths = ga.empty( cl_queue, self.nphotons*ncopies, dtype=np.float32)
            self.t = ga.zeros( cl_queue, self.nphotons*ncopies, dtype=np.float32)
            self.last_hit_triangles = ga.empty( cl_queue, self.nphotons*ncopies, dtype=np.int32)
            self.flags = ga.empty( cl_queue, self.nphotons*ncopies, dtype=np.uint32)
            self.weights = ga.empty( cl_queue, self.nphotons*ncopies, dtype=np.float32)
        
        self.step_pos1_gpu.set( to_float3( self.steps_array[:,0:3] ) )
        self.step_pos2_gpu.set( to_float3( self.steps_array[:,3:6] ) )
        self.ncopies = ncopies
        self.true_nphotons = self.nphotons

        if self.ncopies!=1:
            raise ValueError('support for multiple copies not supported')

        if api.is_gpu_api_cuda():
            self.gpumod = get_module( "gen_photon_from_step.cu", options=api_options, include_source_directory=True )
        elif api.is_gpu_api_opencl():
            self.gpumod = get_module( "gen_photon_from_step.cl", cl_context, options=api_options, include_source_directory=True )
        self.gpufuncs = GPUFuncs( self.gpumod )
        print "gen photon mem alloc/transfer time=",time.time()-tstart_transfer

        # need random numbers
        tgpu = time.time()
        if seed==None:
            seed = 5
        rng_states = get_rng_states(nthreads_per_block*max_blocks, seed=seed, cl_context=cl_context)
        for first_photon, photons_this_round, blocks in chunk_iterator(self.nphotons, nthreads_per_block, max_blocks):
            if api.is_gpu_api_cuda():
                self.gpufuncs.gen_photon_from_step( np.int32(first_photon), np.int32(self.nphotons), self.source_step_index_gpu,
                                                    self.step_pos1_gpu, self.step_pos2_gpu, self.step_fsratio_gpu,
                                                    np.float32( self.steps_array[0,self._fconst] ), np.float32( self.steps_array[0,self._sconst]  ), np.float32( 128.0 ),
                                                    rng_states,
                                                    self.pos, self.dir, self.pol, self.t, self.wavelengths, self.last_hit_triangles, self.flags, self.weights,
                                                    block=(nthreads_per_block,1,1), grid=(blocks, 1) )
            elif api.is_gpu_api_opencl():
                self.gpufuncs.gen_photon_from_step( cl_queue, ( photons_this_round, 1, 1), None,
                                                    np.int32(first_photon), np.int32(self.nphotons), self.source_step_index_gpu.data,
                                                    self.step_pos1_gpu.data, self.step_pos2_gpu.data, self.step_fsratio_gpu.data,
                                                    np.float32( self.steps_array[0,self._fconst] ), np.float32( self.steps_array[0,self._sconst]  ), np.float32( 128.0 ),
                                                    rng_states.data,
                                                    self.pos.data, self.dir.data, self.pol.data, self.t.data, self.wavelengths.data, 
                                                    self.last_hit_triangles.data, self.flags.data, self.weights.data, g_times_l=False ).wait()
                                                    
            else:
                raise RuntimeError("GPU API is neither CUDA nor OpenCL!")
        if api.is_gpu_api_cuda():
            cuda.Context.get_current().synchronize()
        tend_genphotons =  time.time()
        print "GPUPhotonFromSteps: time to gen photons ",tend_genphotons-tstart_genphotons," secs (gpu time=",time.time()-tgpu,")"

        # Now load modules
        if api.is_gpu_api_cuda():
            self.module = get_module('propagate.cu', options=api_options, include_source_directory=True)
        elif  api.is_gpu_api_opencl():
            self.module = get_module('propagate.cl', cl_context, options=api_options, include_source_directory=True)
        # define the texture references
        self.define_texture_references()
        # get kernel functions
        self.gpu_funcs = GPUFuncs(self.module)

        
    def get(self):
        if api.is_gpu_api_cuda():
            pos  = self.pos.get().ravel().view(np.float32).reshape( self.nphotons, 3 )
            pdir = self.dir.get().ravel().view(np.float32).reshape( self.nphotons, 3 )
            pol  = self.pol.get().ravel().view(np.float32).reshape( self.nphotons, 3 )
        elif api.is_gpu_api_opencl():
            # need to remove the padding from vectors
            pos  = np.zeros( shape=(self.nphotons,3), dtype=np.float32 )
            pdir = np.zeros( shape=(self.nphotons,3), dtype=np.float32 )
            pol  = np.zeros( shape=(self.nphotons,3), dtype=np.float32 )
        gapos = self.pos.get()
        gadir = self.dir.get()
        gapol = self.pol.get()
        for n in xrange(0,self.nphotons):
            for i in xrange(0,3):
                pos[n,i]  = gapos[n][i]
                pdir[n,i] = gadir[n][i]
                pol[n,i]  = gapol[n][i]
        t = self.t.get().view(np.float32)
        wavelengths = self.wavelengths.get().view(np.float32)

        return chroma.event.Photons( pos=pos, dir=pdir, pol=pol, t=t, wavelengths=wavelengths )
