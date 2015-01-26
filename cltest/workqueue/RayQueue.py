import numpy as np
import chroma.api as api
if api.is_gpu_api_opencl():
    import pyopencl as cl
elif api.is_gpu_api_cuda():
    import pycuda.driver as cu
from chroma.gpu.photon import GPUPhotons
from workqueue.queueCheckNode import queueCheckNode

class RayQueue:

    def __init__(self,  context ):
        self.context = context
        self.checknodes = queueCheckNode( context, 0 )

    def simulate(self, photons, sim):
        self.gpuphotons = GPUPhotons( photons, cl_context=self.context )

        self.set_initial_workrequest( 0 )

        self.checknodes.launchOnce( self.gpuphotons, sim )

    def set_initial_workrequest(self, requestid ):
        self.gpuphotons.requested_workcode.set( np.tile( np.uint32(0), len(self.gpuphotons) ) )
        
        
