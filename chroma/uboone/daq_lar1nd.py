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
from chroma.uboone.daq_hist import GPUDAQHist
from chroma import event


class GPUDaqLAr1ND(GPUDAQHist):
    """ DAQ that stores histogram of photon hits."""
    NTDC = None
    NS_PER_TDC = None
    def __init__(self, gpu_detector, ntdcs=None, ns_per_tdc=None, adc_bits=None, ndaq=1, cl_context=None, cl_queue=None ):
        """constructor.
        
        Args:
          gpu_detector: GPUDetector
        Keywords:
          ntdcs: int
            number of time bins per channel
            if not supplied, using class variable value
          ns_per_tdc: float
            nanoseconds per time bin
            if not supplied, using class variable value
          adc_bits:  int
            number of ADC bits (not used yet)
          ndaq: int
            number of daqs
          cl_context: pyopencl.Context
          cl_queue: pyopencl.CommandQueue
        Raises:
          ValueError when ntdcs and ns_per_tdc are found to be NoneType
        """
        if ntdcs==None:
            self.ntdcs = GPUDaqLAr1ND.NTDC
        if ns_per_tdc==None:
            self.ns_per_tdc = GPUDaqLAr1ND.NS_PER_TDC
        super( GPUDaqLAr1ND, self ).__init__( gpu_detector, ntdcs=self.ntdcs, ns_per_tdc=self.ns_per_tdc, 
                                              adc_bits=adc_bits, ndaq=ndaq, cl_context=cl_context, cl_queue=cl_queue )
        if self.ntdcs==None:
            raise ValueError("GPUDaqLAr1ND.NTDC has not been set.")
        if self.ns_per_tdc==None:
            raise ValueError("GPUDaqLAr1ND.NS_PER_TDC has not been set.")

        kernel_filepath = os.path.dirname(os.path.realpath(__file__)) + "/daq_lar1nd"
        if api.is_gpu_api_cuda():
            self.module = cutools.get_cu_module(kernel_filepath+".cu", options=api_options, include_source_directory=True)
        elif api.is_gpu_api_opencl():
            self.module                = cltools.get_cl_module(kernel_filepath+'.cl', cl_context, options=api_options, include_source_directory=True)
        else:
            raise RuntimeError("GPU API is neither CUDA nor OpenCL")

        self.gpu_funcs                     = GPUFuncs(self.module)
        
    def acquire(self, gpuphotons, rng_states, nthreads_per_block=64, max_blocks=1024, start_photon=None, nphotons=None, weight=1.0, cl_context=None):
        """run UBooNE DAQ acquire kernels"""
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
                    self.gpu_funcs.run_daq( comqueue, (photons_this_round,1,1), None,
                                            rng_states.data,
                                            np.uint32(0x1 << 2), np.int32(start_photon+first_photon), np.int32(nphotons),
                                            gpuphotons.t.data,  gpuphotons.pos.data,
                                            gpuphotons.flags.data, gpuphotons.last_hit_triangles.data, gpuphotons.weights.data,
                                            self.solid_id_map_gpu.data,
                                            # -- Detector struct --
                                            self.solid_id_to_channel_index_gpu.data,
                                            # ---------------------
                                            self.uint_adc_gpu.data, np.int32(self.nchannels), np.int32(self.ntdcs), np.float32(self.ns_per_tdc), np.float32(100.0),
                                            self.channel_history_gpu.data, 
                                            # -- Channel transforms --
                                            self.channel_inverse_rot_gpu.data, self.channel_inverse_trans_gpu.data,
                                            # ------------------------
                                            np.float32(weight),
                                            g_times_l=False ).wait()
            # if opencl, need to convert ADC from uint to float
            if api.is_gpu_api_opencl():
                self.gpu_funcs.convert_adc( comqueue, (int(self.nchannels),1,1), None,
                                            self.uint_adc_gpu.data, self.adc_gpu.data, 
                                            np.int32(self.nchannels), np.int32(self.ntdcs), 
                                            g_times_l=False ).wait()

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
        """collect daq info and make GPUChannels instance.
        
        Args:
          nthreads_per_block: int
          cl_context: pyopenc.Context
        Returns:
          GPUChannels
        """
        if api.is_gpu_api_cuda():
            self.earliest_time_gpu  = ga.zeros( self.nchannels, dtype=np.float32 )
            nblocks = int(self.nchannels/nthreads_per_block) + 1
            self.gpu_funcs.get_earliest_hit_time( np.int32(self.nchannels), np.int32(self.ntdcs), np.float32(self.ns_per_tdc),
                                                  self.adc_gpu, self.channel_history_gpu, 
                                                  self.earliest_time_gpu,
                                                  block=(1000,1,1), grid=(1,1) )
            self.adc_gpu.get()
        elif  api.is_gpu_api_opencl():
            comqueue = cl.CommandQueue( cl_context )
            self.earliest_time_gpu = ga.zeros( comqueue, self.nchannels, dtype=np.float32 )
            self.gpu_funcs.get_earliest_hit_time( comqueue, (int(self.nchannels),1,1), None,
                                                  np.int32(self.nchannels), np.int32(self.ntdcs), np.float32(self.ns_per_tdc),
                                                  self.adc_gpu.data, self.channel_history_gpu.data,
                                                  self.earliest_time_gpu.data ).wait()
            self.adc_gpu.get()

        return GPUChannels(self.earliest_time_gpu, self.adc_gpu, self.channel_history_gpu, self.ndaq, self.stride)

    @classmethod
    def build_daq(cls, gpu_geometry, cl_context=None, cl_queue=None):
        """factory method.

        will be called by chroma.Simulation to build DAQ instance.
        Returns:
          GPUDaqLAr1ND instance
        """
        return GPUDaqLAr1ND( gpu_geometry, cl_context=cl_context, cl_queue=cl_queue )

