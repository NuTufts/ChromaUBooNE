#from enum import Enum

class _ApiSingleton(type):
    """Define how the Api singleton gets created. Only allows an instance to be created once."""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_ApiSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ApiSingleton(_ApiSingleton('ApiSingletonMeta',(object,),{})): pass
    

#class Apichoices(Enum):
#    cuda = 0
#    opencl = 1

class _cuda_t:
    def __str__(self):
        return "cuda"
class _opencl_t:
    def __str__(self):
        return "opencl"

class Api(ApiSingleton):
    # we can only create this class once.
    # when created, it determines with we are going to use opencl or cuda. user can specify preference if both are found
    #cuda = Apichoices.cuda
    #opencl = Apichoices.opencl
    cuda = _cuda_t()
    opencl = _opencl_t()
    def __init__( self, apipreference=None ):
        super(Api,self).__init__()
        print "api preference: ",apipreference
        if apipreference!=None and apipreference not in [Api.cuda,Api.opencl]:
            raise ValueError( "invalid prefence for GPU API: ",apipreference )
        try:
            import pycuda.drv as cuda
            self.has_pycuda = True
        except:
            self.has_pycuda = False
        
        try:
            import pyopencl as cl
            self.has_pyopencl = True
        except:
            self.has_pyopencl = False

        if self.has_pycuda and not self.has_pyopecl:
            self.using = Api.cuda
        elif self.has_pyopencl and not self.has_pycuda:
            self.using = Api.opencl
        else:
            if apipreference==None:
                print "Found both APIs. Need preference."
                raise
            self.using = apipreference

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
