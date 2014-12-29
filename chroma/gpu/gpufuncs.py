import chroma.api as gpuapi

class GPUFuncs(object):
    """Simple container class for GPU functions as attributes."""
    def __init__(self, module):
        self.module = module
        self.funcs = {}

    def __getattr__(self, name):
        try:
            return self.funcs[name]
        except KeyError:
            # find and then store function name on the demand
            if gpuapi.is_gpu_api_cuda():
                f = self.module.get_function(name)
                self.funcs[name] = f
                return f
            elif gpuapi.is_gpu_api_opencl():
                f = self.module.__getattr__(name)
                self.funcs[name] = f
                return f
