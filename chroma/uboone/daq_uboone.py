import numpy as np
import os

import chroma.api as api
if api.is_gpu_api_cuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray as ga
    from pycuda import characterize
    import chroma.gpu.cutools as cutools
elif api.is_gpu_api_opencl():
    import pyopencl as cl
    import pyopencl.array as ga
    import chroma.gpu.cltools as cltools
else:
    raise RuntimeError('API neither CUDA or OpenCL')

from chroma.gpu.tools import chunk_iterator, api_options
from chroma.gpu.gpufuncs import GPUFuncs
from chroma.gpu.daq import GPUChannels
from chroma import event


class GPUDaqUBooNE(object):
    # This DAQ expands on the default Chroma DAQ.
    # We collect the time distribution of 
    NTDC = None
    NS_PER_TDC = None
    def __init__(self, gpu_detector, ntdcs=None, ns_per_tdc=None, adc_bits=None, ndaq=1, cl_context=None, cl_queue=None ):
        # ntdcs: number of tdc bins
        # ns_per_tdc = nanoseconds per TDC bin
        # adc_bits does nothing for now. there to 
        if ntdcs==None:
            self.ntdcs = GPUDaqUBooNE.NTDC
        if ns_per_tdc==None:
            self.ns_per_tdc = GPUDaqUBooNE.NS_PER_TDC
        if self.ntdcs==None:
            raise ValueError("GPUDaqUBooNE.NTDC has not been set.")
        if self.ns_per_tdc==None:
            raise ValueError("GPUDaqUBooNE.NS_PER_TDC has not been set.")

        cufilepath = os.path.dirname(os.path.realpath(__file__)) + "/daq_uboone"
        if api.is_gpu_api_cuda():
            self.adc_gpu = ga.zeros( gpu_detector.nchannels*ndaq*self.ntdcs, dtype=np.float32 )      # number of hits in bin (or weight of hits in bin)
            self.channel_history_gpu = ga.zeros( gpu_detector.nchannels, dtype=np.uint32 )
            self.detector_gpu = gpu_detector.detector_gpu  # detector struct
            self.module = cutools.get_cu_module(cufilepath+".cu", options=api_options, include_source_directory=True)
        elif api.is_gpu_api_opencl():
            raise RunTimeError("Haven't built opencl daq")
            if cl_queue==None and cl_context==None:
                raise RuntimeError("OpenCL requires either queue or context to be passed")
            if cl_queue==None:
                cl_queue = cl_context.CommandQueue()
            self.adc_gpu     = ga.zeros(cl_queue, ntdcs*gpu_detector.nchannels*ndaq, dtype=np.float32)
            self.channel_history_gpu = ga.zeros( cl_queue, gpu_detector.nchannels, dtype=np.uint32 )
            self.detector_gpu          = gpu_detector # struct not made in opencl mode, so we keep a copy of the class
            self.module                = cltools.get_cl_module('daq_uboone.cl', cl_context, options=api_options, include_source_directory=True)
        else:
            raise RuntimeError("GPU API is neither CUDA nor OpenCL")

        self.solid_id_map_gpu              = gpu_detector.solid_id_map
        self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu
        self.gpu_funcs                     = GPUFuncs(self.module)
        self.ndaq                          = ndaq
        self.stride                        = gpu_detector.nchannels*self.ntdcs # stride between daqs
        self.nchannels                     = gpu_detector.nchannels

    def begin_acquire(self, nthreads_per_block=64, cl_context=None):
        if api.is_gpu_api_cuda():
            self.adc_gpu.fill(0)
            self.channel_history_gpu.fill(0)
        elif api.is_gpu_api_opencl():
            comqueue = cl.CommandQueue(cl_context)
            self.adc_gpu.fill(0.0,queue=comqueue)
            self.channel_history_gpu.fill(0,queue=comqueue)
            cl.enqueue_barrier(comqueue)
        

    def acquire(self, gpuphotons, rng_states, nthreads_per_block=64, max_blocks=1024, start_photon=None, nphotons=None, weight=1.0, cl_context=None):
        if start_photon is None:
            start_photon = 0
        if nphotons is None:
            nphotons = len(gpuphotons.pos) - start_photon

        if api.is_gpu_api_opencl():
            comqueue = cl.CommandQueue( cl_context )
            clmaxblocks = max_blocks

        # We loop over all photons and bin them essentially
        if self.ndaq == 1:
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, nthreads_per_block, max_blocks):
                if api.is_gpu_api_cuda():
                    self.gpu_funcs.run_daq(rng_states, np.uint32(event.SURFACE_DETECT), 
                                           np.int32(start_photon+first_photon), np.int32(photons_this_round), gpuphotons.t, 
                                           gpuphotons.flags, gpuphotons.last_hit_triangles, gpuphotons.weights,
                                           self.solid_id_map_gpu,
                                           self.detector_gpu,
                                           self.adc_gpu, np.int32(self.nchannels), np.int32(self.ntdcs), np.float32(self.ns_per_tdc), np.float32(100.0),
                                           self.channel_history_gpu,
                                           np.float32(weight),
                                           block=(nthreads_per_block,1,1), grid=(blocks,1))
                elif api.is_gpu_api_opencl():
                    raise RunTimeError("Haven't built opencl daq")
                    self.gpu_funcs.run_daq( comqueue, (nthreads_per_block,1,1), (blocks,1),
                                            rng_states.data,
                                            np.uint32(0x1 << 2), np.int32(start_photon+first_photon), np.int32(photons_this_round),
                                            gpuphotons.t.data,  gpuphotons.flags.data, gpuphotons.last_hit_triangles.data, gpuphotons.weights.data,
                                            self.solid_id_map_gpu.data,
                                            # -- Detector struct --
                                            self.solid_id_to_channel_index_gpu.data,
                                            self.detector_gpu.time_cdf_x_gpu.data, self.detector_gpu.time_cdf_y_gpu.data,
                                            self.detector_gpu.charge_cdf_x_gpu.data, self.detector_gpu.charge_cdf_y_gpu.data,
                                            self.detector_gpu.nchannels, self.detector_gpu.time_cdf_len, self.detector_gpu.charge_cdf_len, self.detector_gpu.charge_unit,
                                            # ---------------------
                                            self.earliest_time_int_gpu.data, self.channel_q_int_gpu.data, self.channel_history_gpu.data, np.float32(weight),
                                            g_times_l=True ).wait()
                                            
        else:
            raise RunTimeError("Multi-DAQ not built")
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, 1, max_blocks):
                if api.is_gpu_api_cuda():
                    self.gpu_funcs.run_daq_many(rng_states, np.uint32(0x1 << 2), 
                                                np.int32(start_photon+first_photon), np.int32(photons_this_round), gpuphotons.t, 
                                                gpuphotons.flags, gpuphotons.last_hit_triangles, gpuphotons.weights,
                                                self.solid_id_map_gpu,
                                                self.detector_gpu,
                                                self.earliest_time_int_gpu, 
                                                self.channel_q_int_gpu, self.channel_history_gpu, 
                                                np.int32(self.ndaq), np.int32(self.stride),
                                                np.float32(weight),
                                                block=(nthreads_per_block,1,1), grid=(blocks,1))
                elif api.is_gpu_api_opencl():
                    self.gpu_funcs.run_daq_many( comqueue, (nthreads_per_block,1,1), (blocks,1),
                                                 np.int32(start_photon+first_photon), np.int32(photons_this_round),
                                                 gpuphotons.t.data, gpuphotons.flags.data, gpuphotons.last_hit_triangles.data, gpuphotons.weights.data,
                                                 self.solid_id_map_gpu,
                                                 # -- Detector Struct --
                                                 self.solid_id_to_channel_index_gpu.data,
                                                 self.detector_gpu.time_cdf_x_gpu.data, self.detector_gpu.time_cdf_y_gpu.data,
                                                 self.detector_gpu.charge_cdf_x_gpu.data, self.detector_gpu.charge_cdf_y_gpu.data,
                                                 self.detector_gpu.nchannels, self.detector_gpu.time_cdf_len, self.detector_gpu.charge_cdf_len, self.detector_gpu.charge_unit,
                                                 # ---------------------
                                                 self.earliest_time_int_gpu.data,  self.channel_q_int_gpu.data, self.channel_history_gpu.data,
                                                 np.int32(self.ndaq), np.int32(self.stride), np.float32(weight),
                                                 g_times_l=True ).wait()
        if api.is_gpu_api_cuda():
            cuda.Context.get_current().synchronize()
        elif api.is_gpu_api_opencl():
            cl.enqueue_barrier(comqueue)

    
    def end_acquire(self, nthreads_per_block=64, cl_context=None ):
        self.earliest_time_gpu  = ga.zeros( self.nchannels, dtype=np.float32 )
        #nblocks = int(self.nchannels/nthreads_per_block) + 1
        nblocks = int(1000/nthreads_per_block) + 1
        if api.is_gpu_api_cuda():
            self.gpu_funcs.get_earliest_hit_time( np.int32(self.nchannels), np.int32(self.ntdcs), np.float32(self.ns_per_tdc),
                                                  self.adc_gpu, self.channel_history_gpu, 
                                                  self.earliest_time_gpu,
                                                  block=(1000,1,1), grid=(1,1) )
            self.adc_gpu.get()
        return GPUChannels(self.earliest_time_gpu, self.adc_gpu, self.channel_history_gpu, self.ndaq, self.stride)

    @classmethod
    def build_daq(cls, gpu_geometry, cl_context=None, cl_queue=None):
        return GPUDaqUBooNE( gpu_geometry, cl_context=cl_context, cl_queue=cl_queue )

