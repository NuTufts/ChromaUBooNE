import chroma.api as api

print api.get_gpu_api()
print api.is_gpu_api_opencl()
print api.is_gpu_api_cuda()
