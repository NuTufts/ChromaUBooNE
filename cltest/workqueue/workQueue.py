import os,sys
import chroma.api as api
if api.is_gpu_api_cuda():
    import pycuda.driver as cuda
elif api.is_gpu_api_opencl():
    import pyopencl as cl
import chroma.gpu.tools as tools

class workQueue(object):

    def __init__(self, context ):
        # we get important information about work queues here
        self.context = context
        if api.is_gpu_api_opencl():
            self.device = context.get_info( cl.context_info.DEVICES )[0]
            self.shared_mem_size = self.device.get_info( cl.device_info.LOCAL_MEM_SIZE )
            self.work_group_size = self.device.get_info( cl.device_info.MAX_WORK_GROUP_SIZE )
            self.work_item_sizes = self.device.get_info( cl.device_info.MAX_WORK_ITEM_SIZES )
            self.work_item_dims  = self.device.get_info( cl.device_info.MAX_WORK_ITEM_DIMENSIONS )
            self.max_compute_units = self.device.get_info( cl.device_info.MAX_COMPUTE_UNITS )
        else:
            raise RuntimeError('oops')
    def print_dev_info(self):
        print self.device, self.shared_mem_size, self.work_group_size, self.work_group_size, self.max_compute_units

if __name__ == "__main__":
    # Testing.
    os.environ['PYOPENCL_CTX']='0:1'
    context = tools.get_context()
    w = workQueue( context )
