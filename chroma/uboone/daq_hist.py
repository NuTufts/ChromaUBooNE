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


class GPUDAQHist(object):
    """ DAQ that stores histogram of photon hits."""
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
        if self.ntdcs==None:
            raise ValueError("GPUDAQHist.NTDC has not been set.")
        if self.ns_per_tdc==None:
            raise ValueError("GPUDAQHist.NS_PER_TDC has not been set.")

        kernel_filepath = os.path.dirname(os.path.realpath(__file__)) + "/daq_uboone"
        if api.is_gpu_api_cuda():
            self.adc_gpu = ga.zeros( gpu_detector.nchannels*ndaq*self.ntdcs, dtype=np.float32 )      # number of hits in bin (or weight of hits in bin)
            self.channel_history_gpu = ga.zeros( gpu_detector.nchannels, dtype=np.uint32 )
            self.detector_gpu = gpu_detector.detector_gpu  # detector struct
            #self.module = cutools.get_cu_module(kernel_filepath+".cu", options=api_options, include_source_directory=True)
        elif api.is_gpu_api_opencl():
            if cl_queue==None and cl_context==None:
                raise RuntimeError("OpenCL requires either queue or context to be passed")
            if cl_queue==None:
                cl_queue = cl_context.CommandQueue()
            self.adc_gpu          = ga.zeros(cl_queue, self.ntdcs*gpu_detector.nchannels*ndaq, dtype=np.float32)
            self.uint_adc_gpu     = ga.zeros(cl_queue, self.ntdcs*gpu_detector.nchannels*ndaq, dtype=np.uint32)
            self.channel_history_gpu = ga.zeros( cl_queue, gpu_detector.nchannels, dtype=np.uint32 )
            self.detector_gpu          = gpu_detector # struct not made in opencl mode, so we keep a copy of the class
            #self.module                = cltools.get_cl_module(kernel_filepath+'.cl', cl_context, options=api_options, include_source_directory=True)
        else:
            raise RuntimeError("GPU API is neither CUDA nor OpenCL")

        self.solid_id_map_gpu              = gpu_detector.solid_id_map
        self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu
        #self.gpu_funcs                     = GPUFuncs(self.module)
        self.ndaq                          = ndaq
        self.stride                        = gpu_detector.nchannels*self.ntdcs # stride between daqs
        self.nchannels                     = gpu_detector.nchannels

        # Get geometry info for channels
        self.channel_inverse_rot, self.channel_inverse_trans, self.channel_centroid =  self._save_detector_geoinfo( gpu_detector.geometry.solids,
                                                                                                                    gpu_detector.geometry.solid_id_to_channel_index, 
                                                                                                                    gpu_detector.geometry.channel_index_to_solid_id, 
                                                                                                                    gpu_detector.geometry.channel_index_to_channel_id,
                                                                                                                    gpu_detector.geometry.channel_id_to_channel_index )
        if api.is_gpu_api_cuda():
            self.channel_inverse_rot_gpu = ga.to_gpu( self.channel_inverse_rot.flatten() )
            self.channel_inverse_trans_gpu = ga.to_gpu( self.channel_inverse_trans.flatten() )
        elif api.is_gpu_api_opencl():
            self.channel_inverse_rot_gpu   = ga.to_device( cl_queue, self.channel_inverse_rot.flatten() )
            self.channel_inverse_trans_gpu = ga.to_device( cl_queue, self.channel_inverse_trans.flatten() )
        
    def begin_acquire(self, nthreads_per_block=64, cl_context=None):
        """clear channel counts"""
        if api.is_gpu_api_cuda():
            self.adc_gpu.fill(0)
            self.channel_history_gpu.fill(0)
        elif api.is_gpu_api_opencl():
            comqueue = cl.CommandQueue(cl_context)
            self.adc_gpu.fill(0.0,queue=comqueue)
            self.channel_history_gpu.fill(0,queue=comqueue)
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

    def _save_detector_geoinfo( self, solids, solid_id_to_channel_index, channel_index_to_solid_id, channel_index_to_channel_id, channel_id_to_channel_index ):
        """get transform information for each channel solid.

        we return geometry information for each channel.  this will be used
        to model detector response that is position dependent. 
        for now centers of detectors will be used to match channels with
        geant detectors.

        Args:
          solid_id_to_channel_index: numpy array
          channel_index_to_solid_id: numpy array
          channel_index_to_channel_id: numpy array
          channel_id_to_channel_index: dict
        Returns:
           channel_inverse_rot: numpy array
             (NChannels,3,3) array containing inverse rotation matrix for each channel
           channel_inverse_trans: numpy array
             (NChannels,3) 3-vector containing inverse translation for each channel
           channel_centroid: numpy array
             (NChannel,3) array containing center of channel volume
        """
        channel_inverse_rot = np.zeros( (self.nchannels,3,3), dtype=np.float )
        channel_inverse_trans = np.zeros( (self.nchannels,3), dtype=np.float )
        channel_centroid = np.zeros( (self.nchannels,3), dtype=np.float )
        
        for ich,solid_index in enumerate(channel_index_to_solid_id.ravel()):
            solid = solids[solid_index]
            # get inverted matrix
            rot_matrix = np.matrix( solid.node.boundgeom.matrix[0:3,0:3] )
            channel_inverse_rot[ich,:,:] = rot_matrix.getI()

            # get inverse translation
            channel_inverse_trans[ich,:] = -solid.node.boundgeom.matrix[0:3,3]

            # get translation
            channel_centroid[ich,:] = solid.node.boundgeom.matrix[0:3,3]

        return channel_inverse_rot, channel_inverse_trans, channel_centroid
            
