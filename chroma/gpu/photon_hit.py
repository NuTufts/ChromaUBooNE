

import logging
log = logging.getLogger(__name__)
import numpy as np
import sys
import gc
from pycuda import gpuarray as ga
import pycuda.driver as cuda

from chroma.tools import profile_if_possible
from chroma import event
from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    chunk_iterator, to_float3

class NPY(np.ndarray):pass

class GPUPhotonsHit(object):
    def __init__(self, photons, ncopies=1, max_time=4.):
        """Load ``photons`` onto the GPU, replicating as requested.

           Args:
               - photons: chroma.Event.Photons
                   Photon state information to load onto GPU
               - ncopies: int, *optional*
                   Number of times to replicate the photons
                   on the GPU.  This is used if you want
                   to propagate the same event many times,
                   for example in a likelihood calculation.

                   The amount of GPU storage will be proportionally
                   larger if ncopies > 1, so be careful.
        """

        module = get_cu_module('propagate_hit.cu', options=cuda_options)
        propagate_hit_kernel = module.get_function('propagate_hit')
        propagate_hit_kernel.prepare('iiPPPPPPPPPPPiiiPPP')
        self.propagate_hit_kernel = propagate_hit_kernel
        self.gpu_funcs = GPUFuncs(module)


        self.max_time = max_time 
        self.ncopies = ncopies
        self.true_nphotons = len(photons)
        self.marshall_photons( photons, ncopies)

    def marshall_photons_npl(self, npl):
        pass


    def marshall_photons(self, photons, ncopies):
        """
        Assign the provided photons to the beginning (possibly
        the entire array if ncopies is 1
        """
        nphotons = len(photons)
        self.pos = ga.empty(shape=nphotons*ncopies, dtype=ga.vec.float3)
        self.dir = ga.empty(shape=nphotons*ncopies, dtype=ga.vec.float3)
        self.pol = ga.empty(shape=nphotons*ncopies, dtype=ga.vec.float3)
        self.wavelengths = ga.empty(shape=nphotons*ncopies, dtype=np.float32)
        self.t = ga.empty(shape=nphotons*ncopies, dtype=np.float32)
        self.last_hit_triangles = ga.empty(shape=nphotons*ncopies, dtype=np.int32)
        self.flags = ga.empty(shape=nphotons*ncopies, dtype=np.uint32)
        self.weights = ga.empty(shape=nphotons*ncopies, dtype=np.float32)

        self.pos[:nphotons].set(to_float3(photons.pos))
        self.dir[:nphotons].set(to_float3(photons.dir))
        self.pol[:nphotons].set(to_float3(photons.pol))
        self.wavelengths[:nphotons].set(photons.wavelengths.astype(np.float32))
        self.t[:nphotons].set(photons.t.astype(np.float32))
        self.last_hit_triangles[:nphotons].set(photons.last_hit_triangles.astype(np.int32))
        self.flags[:nphotons].set(photons.flags.astype(np.uint32))
        self.weights[:nphotons].set(photons.weights.astype(np.float32))

        # Replicate the photons to the rest of the slots if needed
        if ncopies > 1:
            max_blocks = 1024
            nthreads_per_block = 64
            block = (nthreads_per_block,1,1)
            for first_photon, photons_this_round, blocks in chunk_iterator(nphotons, nthreads_per_block, max_blocks):
                pass
                grid = (blocks, 1)
                args = (
                        np.int32(first_photon), 
                        np.int32(photons_this_round),
                        self.pos, 
                        self.dir, 
                        self.wavelengths, 
                        self.pol, 
                        self.t, 
                        self.flags, 
                        self.last_hit_triangles, 
                        self.weights,
                        np.int32(ncopies-1), 
                        np.int32(nphotons),
                       )
                self.gpu_funcs.photon_duplicate( *args, block=block, grid=grid)
            pass
        pass

    def get(self, npl=0, hit=0):
        log.info("get npl:%d hit:%d " % (npl, hit)) 
        pos = self.pos.get().view(np.float32).reshape((len(self.pos),3))
        dir = self.dir.get().view(np.float32).reshape((len(self.dir),3))
        pol = self.pol.get().view(np.float32).reshape((len(self.pol),3))
        wavelengths = self.wavelengths.get()
        t = self.t.get()
        last_hit_triangles = self.last_hit_triangles.get()
        flags = self.flags.get()
        weights = self.weights.get()

        if npl:
            nall = len(pos)
            a = np.zeros( (nall,4,4), dtype=np.float32 )       
     
            a[:,0,:3] = pos
            a[:,0, 3] = t 

            a[:,1,:3] = dir
            a[:,1, 3] = wavelengths

            a[:,2,:3] = pol
            a[:,2, 3] = weights 

            assert len(last_hit_triangles) == len(flags)
            pmtid = np.zeros( nall, dtype=np.int32 )

            # a kludge setting of pmtid into lht using the map argument of propagate_hit.cu 
            SURFACE_DETECT = 0x1 << 2
            detected = np.where( flags & SURFACE_DETECT  )
            pmtid[detected] = last_hit_triangles[detected]      # sparsely populate, leaving zeros for undetected

            a[:,3, 0] = np.arange(nall, dtype=np.int32).view(a.dtype)  # photon_id
            a[:,3, 1] = 0                                              # used in comparison againt vbo prop
            a[:,3, 2] = flags.view(a.dtype)                            # history flags 
            a[:,3, 3] = pmtid.view(a.dtype)                            # channel_id ie PmtId

            if hit:
                return a[pmtid > 0].view(NPY)
            else:
                return a.view(NPY)  
            pass
        else:           # the old way 
            return event.Photons(pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights)



    def iterate_copies(self):
        '''Returns an iterator that yields GPUPhotonsSlice objects
        corresponding to the event copies stored in ``self``.'''
        for i in xrange(self.ncopies):
            window = slice(self.true_nphotons*i, self.true_nphotons*(i+1))
            yield GPUPhotonsSlice(pos=self.pos[window],
                                  dir=self.dir[window],
                                  pol=self.pol[window],
                                  wavelengths=self.wavelengths[window],
                                  t=self.t[window],
                                  last_hit_triangles=self.last_hit_triangles[window],
                                  flags=self.flags[window],
                                  weights=self.weights[window])

    def upload_queues(self, nwork):
        """
        # Order photons initially in the queue to put the clones next to each other


        #. input_queue starts as [0,0,1,2,3,.....,nwork]

        #. output_queue starts as [1,0,0,0,0,....] 
 
        """
        input_queue = np.empty(shape=nwork+1, dtype=np.uint32)
        input_queue[0] = 0

        for copy in xrange(self.ncopies):
            input_queue[1+copy::self.ncopies] = np.arange(self.true_nphotons, dtype=np.uint32) + copy * self.true_nphotons

        output_queue = np.zeros(shape=nwork+1, dtype=np.uint32)
        output_queue[0] = 1

        self.input_queue_gpu = ga.to_gpu(input_queue)
        self.output_queue_gpu = ga.to_gpu(output_queue)

    def swap_queues(self):
        """ 
        Swaps queues and returns photons remaining to propagate

        #. output_queue[0] = 1 initially, this avoids enqueued photon_id 
           stomping on output_queue[0] as atomicAdd returns the non-incremented::

                230     // Not done, put photon in output queue
                231     if ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)) == 0) 
                232     {  
                            // pulling queue ticket
                233         int out_idx = atomicAdd(output_queue, 1);  // atomic add 1 to slot zero value, returns non-incremented original value
                234         output_queue[out_idx] = photon_id;
                235     }


        #. At kernel tail non-completed photon threads enqueue their photon_id
           into a slot in the output_queue. The slot to use is obtained 
           by atomic incrementing output_queue[0], ensuring orderly queue.

        #. after kernel completes output_queue[0] contains the
           number of photon_id enqued in output_queue[1:] 

        """
        temp = self.input_queue_gpu
        self.input_queue_gpu = self.output_queue_gpu
        self.output_queue_gpu = temp
        self.output_queue_gpu[:1].set(np.ones(shape=1, dtype=np.uint32))  
        slot0minus1 = self.input_queue_gpu[:1].get()[0] - 1  # which was just now the output_queue before swap
        log.debug("swap_queues slot0minus1 %s " % slot0minus1 )
        return slot0minus1


    @profile_if_possible
    def propagate_hit(self, 
                  gpu_geometry, 
                  rng_states, 
                  parameters):

        """Propagate photons on GPU to termination or max_steps, whichever
        comes first.

        May be called repeatedly without reloading photon information if
        single-stepping through photon history.

        ..warning::
            `rng_states` must have at least `nthreads_per_block`*`max_blocks`
            number of curandStates.


        got one abort::

             In [1]: a = ph("hhMOCK")

             In [9]: f = a[:,3,2].view(np.uint32)

             In [12]: np.where( f & 1<<31 )
             Out[12]: (array([279]),)

        failed to just mock that one::

              RANGE=279:280 MockNuWa MOCK 


        """
        nphotons = self.pos.size
        nwork = nphotons

        nthreads_per_block = parameters['threads_per_block'] 
        max_blocks = parameters['max_blocks'] 
        max_steps = parameters['max_steps'] 
        use_weights=False
        scatter_first=0

        self.upload_queues( nwork )

        solid_id_map_gpu = gpu_geometry.solid_id_map
        solid_id_to_channel_id_gpu = gpu_geometry.solid_id_to_channel_id_gpu

        small_remainder = nthreads_per_block * 16 * 8 
        block=(nthreads_per_block,1,1)

        results = {}
        results['name'] = "propagate_hit"
        results['nphotons'] = nphotons
        results['nwork'] = nwork
        results['nsmall'] = small_remainder
        results['COLUMNS'] = "name:s,nphotons:i,nwork:i,nsmall:i"

        step = 0
        times = []

        npass = 0  
        nabort = 0 

        while step < max_steps:
            npass += 1
            if nwork < small_remainder or use_weights:
                nsteps = max_steps - step # Just finish the rest of the steps if the # of photons is low
                log.debug("increase nsteps for stragglers: small_remainder %s nwork %s nsteps %s max_steps %s " % (small_remainder, nwork, nsteps, max_steps))
            else:
                nsteps = 1 
            pass
            log.info("nphotons %s nwork %s step %s max_steps %s nsteps %s " % (nphotons, nwork, step,max_steps, nsteps) )

            abort = False
            for first_photon, photons_this_round, blocks in chunk_iterator(nwork, nthreads_per_block, max_blocks):
                if abort:
                    nabort += 1   
                else:
                    grid = (blocks, 1)
                    args = (
                        np.int32(first_photon), 
                        np.int32(photons_this_round), 
                        self.input_queue_gpu[1:].gpudata, 
                        self.output_queue_gpu.gpudata, 
                        rng_states, 
                        self.pos.gpudata, 
                        self.dir.gpudata, 
                        self.wavelengths.gpudata, 
                        self.pol.gpudata, 
                        self.t.gpudata, 
                        self.flags.gpudata, 
                        self.last_hit_triangles.gpudata, 
                        self.weights.gpudata, 
                        np.int32(nsteps), 
                        np.int32(use_weights), 
                        np.int32(scatter_first), 
                        gpu_geometry.gpudata, 
                        solid_id_map_gpu.gpudata,
                        solid_id_to_channel_id_gpu.gpudata,
                            )

                    log.info("propagate_hit_kernel.prepared_timed_call grid %s block %s first_photon %s photons_this_round %s " % (repr(grid), repr(block), first_photon, photons_this_round))
                    get_time = self.propagate_hit_kernel.prepared_timed_call( grid, block, *args )
                    t = get_time()
                    times.append( t )
                    if t > self.max_time:
                        abort = True
                        log.warn("kernel launch time %s > max_time %s : ABORTING " % (t, self.max_time) )
                    pass
                pass
            pass
            log.info("step %s propagate_hit_kernel times  %s " % (step, repr(times)) )
            pass
            step += nsteps
            scatter_first = 0 # Only allow non-zero in first pass
            if step < max_steps:
                nwork = self.swap_queues()
            pass
        pass

        log.info("calling max ")
        if ga.max(self.flags).get() & (1 << 31):
            log.warn("ABORTED PHOTONS")
        log.info("done calling max ")

        cuda.Context.get_current().synchronize()

        results['npass'] = npass
        results['nabort'] = nabort
        results['nlaunch'] = len(times)
        results['tottime'] = sum(times)
        results['maxtime'] = max(times)
        results['mintime'] = min(times)
        results['COLUMNS'] += ",npass:i,nabort:i,nlaunch:i,tottime:f,maxtime:f,mintime:f"  
        return results 


    @profile_if_possible
    def select(self, target_flag, nthreads_per_block=64, max_blocks=1024,
               start_photon=None, nphotons=None):
        '''Return a new GPUPhoton object containing only photons that
        have a particular bit set in their history word.'''
        cuda.Context.get_current().synchronize()
        index_counter_gpu = ga.zeros(shape=1, dtype=np.uint32)
        cuda.Context.get_current().synchronize()
        if start_photon is None:
            start_photon = 0
        if nphotons is None:
            nphotons = self.pos.size - start_photon

        # First count how much space we need
        for first_photon, photons_this_round, blocks in \
                chunk_iterator(nphotons, nthreads_per_block, max_blocks):
            self.gpu_funcs.count_photons(np.int32(start_photon+first_photon), 
                                         np.int32(photons_this_round),
                                         np.uint32(target_flag),
                                         index_counter_gpu, self.flags,
                                         block=(nthreads_per_block,1,1), 
                                         grid=(blocks, 1))
        cuda.Context.get_current().synchronize()
        reduced_nphotons = int(index_counter_gpu.get()[0])
        # Then allocate new storage space
        pos = ga.empty(shape=reduced_nphotons, dtype=ga.vec.float3)
        dir = ga.empty(shape=reduced_nphotons, dtype=ga.vec.float3)
        pol = ga.empty(shape=reduced_nphotons, dtype=ga.vec.float3)
        wavelengths = ga.empty(shape=reduced_nphotons, dtype=np.float32)
        t = ga.empty(shape=reduced_nphotons, dtype=np.float32)
        last_hit_triangles = ga.empty(shape=reduced_nphotons, dtype=np.int32)
        flags = ga.empty(shape=reduced_nphotons, dtype=np.uint32)
        weights = ga.empty(shape=reduced_nphotons, dtype=np.float32)

        # And finaly copy photons, if there are any
        if reduced_nphotons > 0:
            index_counter_gpu.fill(0)
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, nthreads_per_block, max_blocks):
                self.gpu_funcs.copy_photons(np.int32(start_photon+first_photon), 
                                            np.int32(photons_this_round), 
                                            np.uint32(target_flag),
                                            index_counter_gpu, 
                                            self.pos, self.dir, self.wavelengths, self.pol, self.t, self.flags, self.last_hit_triangles, self.weights,
                                            pos, dir, wavelengths, pol, t, flags, last_hit_triangles, weights,
                                            block=(nthreads_per_block,1,1), 
                                            grid=(blocks, 1))
            assert index_counter_gpu.get()[0] == reduced_nphotons
        return GPUPhotonsHitSlice(pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights)

    def __del__(self):
        del self.pos
        del self.dir
        del self.pol
        del self.wavelengths
        del self.t
        del self.flags
        del self.last_hit_triangles
        # Free up GPU memory quickly if now available
        gc.collect()


    def __len__(self):
        return self.pos.size

class GPUPhotonsHitSlice(GPUPhotonsHit):
    '''A `slice`-like view of a subrange of another GPU photons array.
    Works exactly like an instance of GPUPhotons, but the GPU storage
    is taken from another GPUPhotons instance.

    Returned by the GPUPhotons.iterate_copies() iterator.'''
    def __init__(self, pos, dir, pol, wavelengths, t, last_hit_triangles,
                 flags, weights):
        '''Create new object using slices of GPUArrays from an instance
        of GPUPhotons.  NOTE THESE ARE NOT CPU ARRAYS!'''
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.wavelengths = wavelengths
        self.t = t
        self.last_hit_triangles = last_hit_triangles
        self.flags = flags
        self.weights = weights

        module = get_cu_module('propagate.cu', options=cuda_options)
        self.gpu_funcs = GPUFuncs(module)

        self.true_nphotons = len(pos)
        self.ncopies = 1

    def __del__(self):
        pass # Do nothing, because we don't own any of our GPU memory
