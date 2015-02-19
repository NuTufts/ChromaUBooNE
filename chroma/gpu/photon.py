import numpy as np
import sys
import gc
import chroma.api as api
if api.is_gpu_api_cuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray as ga
elif api.is_gpu_api_opencl():
    import pyopencl as cl
    #from pyopencl.array import Array as ga
    import pyopencl.array as ga
from chroma.tools import profile_if_possible
from chroma import event
from chroma.gpu.tools import get_module, api_options, chunk_iterator, to_float3, copy_to_float3
from chroma.gpu.gpufuncs import GPUFuncs
import time


class GPUPhotons(object):
    def __init__(self, photons, ncopies=1, cl_context=None):
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
        nphotons = len(photons)
        # Allocate GPU memory for photon info and push to device
        if api.is_gpu_api_cuda():
            self.pos = ga.empty(shape=nphotons*ncopies, dtype=ga.vec.float3)
            self.dir = ga.empty(shape=nphotons*ncopies, dtype=ga.vec.float3)
            self.pol = ga.empty(shape=nphotons*ncopies, dtype=ga.vec.float3)
            self.wavelengths = ga.empty(shape=nphotons*ncopies, dtype=np.float32)
            self.t = ga.empty(shape=nphotons*ncopies, dtype=np.float32)
            self.last_hit_triangles = ga.empty(shape=nphotons*ncopies, dtype=np.int32)
            self.flags = ga.empty(shape=nphotons*ncopies, dtype=np.uint32)
            self.weights = ga.empty(shape=nphotons*ncopies, dtype=np.float32)
            self.current_node_index = ga.zeros( shape=nphotons*ncopies, dtype=np.uint32 ) # deprecated
            self.requested_workcode = ga.empty( shape=nphotons*ncopies, dtype=np.uint32 ) # deprecated
        elif api.is_gpu_api_opencl():
            queue = cl.CommandQueue( cl_context )
            self.pos = ga.empty(queue, shape=nphotons*ncopies, dtype=ga.vec.float3)
            self.dir = ga.empty(queue, shape=nphotons*ncopies, dtype=ga.vec.float3)
            self.pol = ga.empty(queue, shape=nphotons*ncopies, dtype=ga.vec.float3)
            self.wavelengths = ga.empty(queue, shape=nphotons*ncopies, dtype=np.float32)
            self.t = ga.empty(queue, shape=nphotons*ncopies, dtype=np.float32)
            self.last_hit_triangles = ga.empty(queue, shape=nphotons*ncopies, dtype=np.int32)
            self.flags = ga.empty(queue, shape=nphotons*ncopies, dtype=np.uint32)
            self.weights = ga.empty(queue, shape=nphotons*ncopies, dtype=np.float32)
            self.current_node_index = ga.zeros( queue, shape=nphotons*ncopies, dtype=np.uint32 ) # deprecated
            self.requested_workcode = ga.empty( queue, shape=nphotons*ncopies, dtype=np.uint32 ) # deprecated

        # Assign the provided photons to the beginning (possibly
        # the entire array if ncopies is 1
        self.pos[:nphotons].set(to_float3(photons.pos))
        self.dir[:nphotons].set(to_float3(photons.dir))
        self.pol[:nphotons].set(to_float3(photons.pol))
        self.wavelengths[:nphotons].set(photons.wavelengths.astype(np.float32))
        self.t[:nphotons].set(photons.t.astype(np.float32))
        self.last_hit_triangles[:nphotons].set(photons.last_hit_triangles.astype(np.int32))
        self.flags[:nphotons].set(photons.flags.astype(np.uint32))
        self.weights[:nphotons].set(photons.weights.astype(np.float32))

        if api.is_gpu_api_cuda():
            self.module = get_module('propagate.cu', options=api_options, include_source_directory=True)
        elif  api.is_gpu_api_opencl():
            self.module = get_module('propagate.cl', cl_context, options=api_options, include_source_directory=True)
        # define the texture references
        self.define_texture_references()
        # get kernel functions
        self.gpu_funcs = GPUFuncs(self.module)

        # Replicate the photons to the rest of the slots if needed
        if ncopies > 1:
            max_blocks = 1024
            nthreads_per_block = 64
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, nthreads_per_block, max_blocks):
                self.gpu_funcs.photon_duplicate(np.int32(first_photon), np.int32(photons_this_round),
                                                self.pos, self.dir, self.wavelengths, self.pol, self.t, 
                                                self.flags, self.last_hit_triangles, self.weights,
                                                np.int32(ncopies-1), 
                                                np.int32(nphotons),
                                                block=(nthreads_per_block,1,1), grid=(blocks, 1))

        # Save the duplication information for the iterate_copies() method
        self.true_nphotons = nphotons
        self.ncopies = ncopies

    def define_texture_references( self, module=None ):
        # unbound texture references declared for use with propagate
        if module==None:
            module = self.module
        if api.is_gpu_api_cuda():
            self.node_texture_ref       = module.get_texref( "nodevec_tex_ref" )
            self.node_texture_ref.set_format( cuda.array_format.UNSIGNED_INT32, 4 )

            self.extra_node_texture_ref = module.get_texref( "extra_node_tex_ref" )
            self.extra_node_texture_ref.set_format( cuda.array_format.UNSIGNED_INT32, 4 )

            self.vertices_texture_ref   = module.get_texref( "verticesvec_tex_ref" )
            self.vertices_texture_ref.set_format( cuda.array_format.FLOAT, 4 )

            self.triangles_texture_ref   = module.get_texref( "trianglesvec_tex_ref" )
            self.triangles_texture_ref.set_format( cuda.array_format.UNSIGNED_INT32, 4 )

            self.node_texture_ref_bound = False
        elif api.is_gpu_api_opencl():
            # texture usage not used at the moment
            pass

    def get(self):
        ncols = 3
        if api.is_gpu_api_opencl():
            ncols = 4 # must include padding
        pos = self.pos.get().view(np.float32).reshape((len(self.pos),ncols))
        dir = self.dir.get().view(np.float32).reshape((len(self.dir),ncols))
        pol = self.pol.get().view(np.float32).reshape((len(self.pol),ncols))
        wavelengths = self.wavelengths.get()
        t = self.t.get()
        last_hit_triangles = self.last_hit_triangles.get()
        flags = self.flags.get()
        weights = self.weights.get()
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

    @profile_if_possible
    def propagate(self, gpu_geometry, rng_states, nthreads_per_block=64,
                  max_blocks=1024, max_steps=10, use_weights=False,
                  scatter_first=0, cl_context=None):
        """Propagate photons on GPU to termination or max_steps, whichever
        comes first.

        May be called repeatedly without reloading photon information if
        single-stepping through photon history.

        ..warning::
            `rng_states` must have at least `nthreads_per_block`*`max_blocks`
            number of curandStates.
        """
        nphotons = self.pos.size
        # bind node texture reference
        if api.is_gpu_api_cuda() and not self.node_texture_ref_bound:
            # we have to unroll, as pycuda doesn't seem to support vector times right now for binding
            self.unrolled_nodes       = ga.to_gpu( gpu_geometry.nodes.get().ravel().view( np.uint32 ) )
            self.unrolled_extra_nodes = ga.to_gpu( gpu_geometry.extra_nodes.ravel().view( np.uint32 ) )
            self.unrolled_triangles   = ga.to_gpu( gpu_geometry.triangles.get().ravel().view( np.uint32 ) )
            self.unrolled_triangles4  = ga.to_gpu( gpu_geometry.triangles4.ravel().view( np.uint32 ) )
            self.unrolled_vertices    = ga.to_gpu( gpu_geometry.vertices.get().ravel().view( np.float32 ) )
            self.unrolled_vertices4   = ga.to_gpu( gpu_geometry.vertices4.ravel().view( np.float32 ) )
            self.node_texture_ref.set_address( self.unrolled_nodes.gpudata, self.unrolled_nodes.nbytes )
            self.extra_node_texture_ref.set_address( self.unrolled_extra_nodes.gpudata, self.unrolled_extra_nodes.nbytes )
            #self.unrolled_nodes.bind_to_texref_ext( self.node_texture_ref )
            #self.unrolled_extra_nodes.bind_to_texref_ext( self.extra_node_texture_ref )
            #self.unrolled_triangles.bind_to_texref_ext( self.triangles_texture_ref )
            self.triangles_texture_ref.set_address( self.unrolled_triangles4.gpudata, self.unrolled_triangles4.nbytes )
            #self.unrolled_vertices.bind_to_texref_ext( self.vertices_texture_ref )
            self.vertices_texture_ref.set_address( self.unrolled_vertices4.gpudata, self.unrolled_vertices4.nbytes )
            print "[BOUND TO TEXTURE MEMORY]"
            print "Nodes: ",self.unrolled_nodes.nbytes/1.0e3," kbytes"
            print "Extra nodes: ",self.unrolled_extra_nodes.nbytes/1.0e3," kbytes"
            print "Triangles: ",self.unrolled_triangles4.nbytes/1.0e3," kbytes"
            print "Vertices: ",self.unrolled_vertices4.nbytes/1.0e3," kbytes"
            print "Total: ",(self.unrolled_nodes.nbytes+self.unrolled_extra_nodes.nbytes+self.unrolled_triangles4.nbytes+self.unrolled_vertices4.nbytes)/1.0e3,"kbytes"
            self.node_texture_ref_bound = True

        # setup queue
        maxqueue = nphotons
        step = 0
        input_queue = np.empty(shape=maxqueue+1, dtype=np.uint32)
        input_queue[0] = 0
        # Order photons initially in the queue to put the clones next to each other
        for copy in xrange(self.ncopies):
            input_queue[1+copy::self.ncopies] = np.arange(self.true_nphotons, dtype=np.uint32) + copy * self.true_nphotons
        if api.is_gpu_api_cuda():
            input_queue_gpu = ga.to_gpu(input_queue)
        elif api.is_gpu_api_opencl():
            comqueue = cl.CommandQueue(cl_context)
            input_queue_gpu = ga.to_device(comqueue,input_queue[1:]) # why the offset?

        output_queue = np.zeros(shape=maxqueue+1, dtype=np.uint32)
        output_queue[0] = 1
        if api.is_gpu_api_cuda():
            output_queue_gpu = ga.to_gpu(output_queue)
        elif api.is_gpu_api_opencl():
            output_queue_gpu = ga.to_device(comqueue,output_queue)

        if use_weights:
            iuse_weights = 1
        else:
            iuse_weights = 0

        adapt_factor = 1.0
        start_prop = time.time()
        while step < max_steps:
            # Just finish the rest of the steps if the # of photons is low
            #if nphotons < nthreads_per_block * 16 * 8 or use_weights:
            #    nsteps = max_steps - step
            #else:
            #    nsteps = 1
            nsteps = 1

            start_step = time.time()
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, nthreads_per_block, max( int(adapt_factor*max_blocks), 1 )):
                #print nphotons, nthreads_per_block, max_blocks," : ",first_photon, photons_this_round, blocks, adapt_factor
                start_chunk = time.time()
                if api.is_gpu_api_cuda():
                    self.gpu_funcs.propagate(np.int32(first_photon), np.int32(photons_this_round), 
                                             input_queue_gpu[1:], output_queue_gpu, rng_states, 
                                             self.pos, self.dir, self.wavelengths, self.pol, self.t, self.flags, self.last_hit_triangles, 
                                             self.weights, np.int32(nsteps), np.int32(iuse_weights), np.int32(scatter_first), 
                                             gpu_geometry.gpudata, block=(nthreads_per_block,1,1), grid=(blocks, 1))
                    cuda.Context.get_current().synchronize()
                elif api.is_gpu_api_opencl():
                    self.gpu_funcs.propagate( comqueue, (photons_this_round,1,1), None,
                                              np.int32(first_photon), np.int32(photons_this_round),
                                              input_queue_gpu.data, output_queue_gpu.data,
                                              rng_states.data, 
                                              self.pos.data, self.dir.data, self.wavelengths.data, self.pol.data, self.t.data, 
                                              self.flags.data, self.last_hit_triangles.data, self.weights.data,
                                              np.int32(nsteps), np.int32(iuse_weights), np.int32(scatter_first),
                                              gpu_geometry.world_scale, gpu_geometry.world_origin.data,  np.int32(len(gpu_geometry.nodes)),
                                              gpu_geometry.material_data['n'], gpu_geometry.material_data['step'], gpu_geometry.material_data["wavelength0"],
                                              gpu_geometry.vertices.data, gpu_geometry.triangles.data,
                                              gpu_geometry.material_codes.data, gpu_geometry.colors.data,
                                              gpu_geometry.nodes.data, gpu_geometry.extra_nodes.data,
                                              gpu_geometry.material_data["nmaterials"],
                                              gpu_geometry.material_data['refractive_index'].data, gpu_geometry.material_data['absorption_length'].data, 
                                              gpu_geometry.material_data['scattering_length'].data, 
                                              gpu_geometry.material_data['reemission_prob'].data, gpu_geometry.material_data['reemission_cdf'].data,
                                              gpu_geometry.surface_data['nsurfaces'],
                                              gpu_geometry.surface_data['detect'].data, gpu_geometry.surface_data['absorb'].data, gpu_geometry.surface_data['reemit'].data,
                                              gpu_geometry.surface_data['reflect_diffuse'].data, gpu_geometry.surface_data['reflect_specular'].data,
                                              gpu_geometry.surface_data['eta'].data, gpu_geometry.surface_data['k'].data, gpu_geometry.surface_data['reemission_cdf'].data,
                                              gpu_geometry.surface_data['model'].data, gpu_geometry.surface_data['transmissive'].data, gpu_geometry.surface_data['thickness'].data,
                                              g_times_l=True ).wait()
                end_chunk = time.time()
                chunk_time = end_chunk-start_chunk
                #print "chunk time: ",chunk_time
                if chunk_time>2.5:
                    adapt_factor *= 0.5
            step += nsteps
            scatter_first = 0 # Only allow non-zero in first pass
            end_step = time.time()
            #print "step time: ",end_step-start_step
            
            if step < max_steps:
                start_requeue = time.time()
                #print "reset photon queues"
                if api.is_gpu_api_cuda():
                    #temp = input_queue_gpu
                    #input_queue_gpu = output_queue_gpu
                    #output_queue_gpu = temp
                    # Assign with a numpy array of length 1 to silence
                    # warning from PyCUDA about setting array with different strides/storage orders.
                    #output_queue_gpu[:1].set(np.ones(shape=1, dtype=np.uint32))
                    #nphotons = input_queue_gpu[:1].get()[0] - 1
                    # new style
                    output_queue_gpu.get( output_queue )
                    nphotons = output_queue[0]-1
                    input_queue_gpu.set( output_queue )
                    output_queue_gpu[:1].set(np.ones(shape=1,dtype=np.uint32))
                    cuda.Context.get_current().synchronize()
                elif api.is_gpu_api_opencl():
                    temp_out = output_queue_gpu.get()
                    nphotons = temp_out[0]
                    input_queue_gpu.set( temp_out[1:], queue=comqueue ) # set the input queue to have index of photons still need to be run
                    output_queue_gpu[:1].set( np.ones(shape=1,dtype=np.uint32), queue=comqueue ) # reset first instance to be one
                end_requeue = time.time()
                #print "re-queue time: ",end_requeue-start_requeue
        end_prop = time.time()
        print "propagation time: ",end_prop-start_prop," secs"
        end_flags = self.flags.get()
        end_flag = np.max(end_flags)
        if end_flag & (1 << 31):
            print >>sys.stderr, "WARNING: ABORTED PHOTONS"
        if api.is_gpu_api_cuda():
            cuda.Context.get_current().synchronize()
        elif api.is_gpu_api_opencl():
            cl.enqueue_barrier( comqueue )

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
        return GPUPhotonsSlice(pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights)

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

class GPUPhotonsSlice(GPUPhotons):
    '''A `slice`-like view of a subrange of another GPU photons array.
    Works exactly like an instance of GPUPhotons, but the GPU storage
    is taken from another GPUPhotons instance.

    Returned by the GPUPhotons.iterate_copies() iterator.'''
    def __init__(self, pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights):
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
