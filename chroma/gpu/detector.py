import numpy as np
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

from chroma.geometry import standard_wavelengths
from chroma.gpu.tools import chunk_iterator, format_array, format_size, to_uint3, to_float3, make_gpu_struct
from chroma.gpu.geometry import GPUGeometry
#from chroma.log import logger
import logging
logger = logging.getLogger(__name__)



class GPUDetector(GPUGeometry):
    def __init__(self, detector, wavelengths=None, print_usage=False):
        GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)

        self.solid_id_to_channel_index_gpu = \
            ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
        self.solid_id_to_channel_id_gpu = \
            ga.to_gpu(detector.solid_id_to_channel_id.astype(np.int32))

        self.nchannels = detector.num_channels()


        self.time_cdf_x_gpu = ga.to_gpu(detector.time_cdf[0].astype(np.float32))
        self.time_cdf_y_gpu = ga.to_gpu(detector.time_cdf[1].astype(np.float32))

        self.charge_cdf_x_gpu = ga.to_gpu(detector.charge_cdf[0].astype(np.float32))
        self.charge_cdf_y_gpu = ga.to_gpu(detector.charge_cdf[1].astype(np.float32))

        detector_source = get_cu_source('detector.h')
        detector_struct_size = characterize.sizeof('Detector', detector_source)
        self.detector_gpu = make_gpu_struct(detector_struct_size,
                                            [self.solid_id_to_channel_index_gpu,
                                             self.time_cdf_x_gpu,
                                             self.time_cdf_y_gpu,
                                             self.charge_cdf_x_gpu,
                                             self.charge_cdf_y_gpu,
                                             np.int32(self.nchannels),
                                             np.int32(len(detector.time_cdf[0])),
                                             np.int32(len(detector.charge_cdf[0])),
                                             np.float32(detector.charge_cdf[0][-1] / 2**16)])
                                             
