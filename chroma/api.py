#from enum import Enum

# This module provides functions to determine what GPU API we are using.
# The Api class stores the choice of Api. It is defined to be a singleton
# so the determination is done only once.
# We have not implemented a way to change the initial choice.  That would seem to open
#   a giant can of worms.

class _ApiSingleton(type):
    """Define how the Api singleton gets created. Only allows an instance to be created once."""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_ApiSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ApiSingleton(_ApiSingleton('ApiSingletonMeta',(object,),{})): pass
    
class _cuda_t:
    def __str__(self):
        return "cuda"
class _opencl_t:
    def __str__(self):
        return "opencl"

class Api(ApiSingleton):
    # we can only create this class once. (the metaclass takes care of that)
    # when created, it determines which API we are going to use: opencl or cuda. user can specify preference if both are found
    #pass in Api.cuda or Api.opencl
    cuda = _cuda_t()
    opencl = _opencl_t()
    def __init__( self, apipreference=None ):
        super(Api,self).__init__()
        if apipreference!=None and apipreference not in [Api.cuda,Api.opencl]:
            raise ValueError( "invalid prefence for GPU API: ",apipreference )
        try:
            import pycuda.driver as cuda
            self.has_pycuda = True
        except:
            self.has_pycuda = False
        
        try:
            import pyopencl as cl
            self.has_pyopencl = True
        except:
            self.has_pyopencl = False
        print "API available OpenCL=",self.has_pyopencl,", CUDA=",self.has_pycuda

        if self.has_pycuda and not self.has_pyopencl:
            # Cuda only
            self.using = Api.cuda
        elif self.has_pyopencl and not self.has_pycuda:
            self.using = Api.opencl
        else:
            if apipreference==None:
                raise ValueError( "Found both APIs. Need preference." )
            self.using = apipreference

        print "Setting API: ",self.using

def get_gpu_api():
    a = Api()
    return a.using

def is_gpu_api_opencl():
    a = Api()
    if a.using==a.opencl:
        return True
    else:
        return False

def is_gpu_api_cuda():
    a = Api()
    if a.using==a.cuda:
        return True
    else:
        return False

def use_cuda():
    a = Api( Api.cuda )

def use_opencl():
    a = Api( Api.opencl )
