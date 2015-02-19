import os
#os.environ['PYOPENCL_CTX']='0:1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'
import chroma.api as api
api.use_cuda()
#api.use_opencl()
if api.is_gpu_api_cuda():
    import pycuda.driver as cuda
    import chroma.gpu.cutools as cutools
elif api.is_gpu_api_opencl():
    import pyopencl as cl
    import chroma.gpu.cltools as cltools
else:
    raise RuntimeError('API not set to either CUDA or OpenCL')
import chroma.gpu.tools as tools
from photon_fromstep import GPUPhotonFromSteps
import numpy as np


context = tools.get_context()

steps = np.load( 'steps.npy' )
photons = GPUPhotonFromSteps( steps[:10,:], cl_context=context )

pos = photons.pos.get()
t   = photons.t.get()
photon_dir = photons.dir.get()
photon_pol = photons.pol.get()
wavelengths = photons.wavelengths.get()

#print pos[400:500]
#print t[400:500]
#print photon_dir[400:500]
#print photon_pol[400:500]
#print wavelengths[400:500]
print steps[0:20,0:3],steps[0:20,0:3]
#for step in steps:
#    print step[0:3],step[3:6]
print pos
print pos.ravel().view(np.float32).reshape( photons.nphotons, 3 )

if api.is_gpu_api_cuda():
    context.pop()
