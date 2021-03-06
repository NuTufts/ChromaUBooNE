# layer attempting to make tools functions API agnostic
import logging
log = logging.getLogger(__name__)
import numpy as np
import chroma.api as gpuapi
if gpuapi.is_gpu_api_cuda():
    import pycuda.driver as cuda
    import pycuda.tools
    import pycuda.gpuarray as ga
    import chroma.gpu.cutools as cutools

    # standard nvcc options
    api_options = cutools.cuda_options

elif gpuapi.is_gpu_api_opencl():
    import pyopencl as cl
    import pyopencl.array as ga
    import chroma.gpu.cltools as cltools

    # standard nvcc options
    api_options = cltools.cl_options


def template_substitute( source, template_uncomment ):
    """
    Substitution is done by removing the instrumented comment, for example
    to swap a metric use `template_uncomment=(("metric","time",))`:: 

        //metric{time}       int64_t metric = clock64() - start ;
        //metric{node}       int     metric = node_count ; 
        //metric{intersect}  int     metric = intersect_count ; 
        //metric{tri}        int     metric = tri_count ; 

    So to use metric=time to get the time line
    """
    for k, v in template_uncomment: 
        marker = "//%s{%s}" % ( k,v )      # eg looking for //metric{time}
        spacer = " " * len(marker)
        source = source.replace(marker, spacer)
        log.debug("replacing marker %s " % marker ) 
    pass
    return source


def template_interpolation( source, template_fill ):
    """  
    :param source:
    :param template_fill: tuple of tuple pairs, eg (("a",1),("b",2))  

    Hashability needed for context_dependent_memoize dictates the type
    """
    return source % dict(template_fill)

# ==========================================
# gpu interface utilities

def get_module(*args, **kwargs):
#def get_module(name, options=None, include_source_directory=True, template_uncomment=None, template_fill=None):
    """ arguments: name, options=None, include_source_directory=True, template_uncomment=None, template_fill=None)"""
    if gpuapi.is_gpu_api_cuda():
        return cutools.get_cu_module( *args, **kwargs )
    elif gpuapi.is_gpu_api_opencl():
        return cltools.get_cl_module( *args, **kwargs )

def get_source(name):
    if gpuapi.is_gpu_api_cuda():
        return cutools.get_cu_source(name)
    elif gpuapi.is_gpu_api_opencl():
        return cltools.get_cl_source(name)

def get_context(*args, **kwargs):
    if gpuapi.is_gpu_api_cuda():
        #return cutools.get_cuda_context(device_id,context_flags)
        return cutools.create_cuda_context(*args, **kwargs)
    elif gpuapi.is_gpu_api_opencl():
        #return cltools.create_cl_context(device_id,context_flags)
        return cltools.create_cl_context(*args, **kwargs)

# ==========================================
# Random number utilities
def get_rng_states(size, seed=1, cl_context=None):
    if gpuapi.is_gpu_api_cuda():
        return cutools.get_rng_states(size,seed=seed)
    elif gpuapi.is_gpu_api_opencl():
        return cltools.get_rng_states(size,seed=seed,cl_context=cl_context)

def get_random_array( size, rng_states ):
    pass


# ==========================================
# vector type utilities
def to_float3(arr):
    "Returns a vec.float3 array from an (N,3) array."
    if gpuapi.is_gpu_api_cuda():
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.asarray(arr, order='c')
        return arr.astype(np.float32).view(ga.vec.float3)[:,0]
    elif gpuapi.is_gpu_api_opencl():
        # in the pyopencl implementation, the vec types have a padding column
        # need to extend this
        n = len(arr)
        pad = np.zeros( (n,1), dtype=arr.dtype )
        arr_wpad = np.hstack( (arr, pad) )
        return arr_wpad.astype(np.float32).view(ga.vec.float3)[:,0]
    else:
        raise RuntimeError('API is neither CUDA nor OpenCL')

def copy_to_float3( arr, f3arr ):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.asarray(arr, order='c')
    f3arr['x'] = arr.astype(np.float32)[:,0]
    f3arr['y'] = arr.astype(np.float32)[:,1]
    f3arr['z'] = arr.astype(np.float32)[:,2]
    f3arr['padding0'] = np.zeros(len(f3arr),dtype=np.float32)[:]

def to_uint3(arr):
    "Returns a vec.uint3 array from an (N,3) array."
    if gpuapi.is_gpu_api_cuda():
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.asarray(arr, order='c')
        return arr.astype(np.uint32).view(ga.vec.uint3)[:,0]
    elif gpuapi.is_gpu_api_opencl():
        n = len(arr)
        pad = np.zeros( (n,1), dtype=arr.dtype )
        arr_wpad = np.hstack( (arr, pad) )
        return arr_wpad.astype(np.uint32).view(ga.vec.uint3)[:,0]

def copy_to_uint3( arr, ui3arr ):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.asarray(arr, order='c')

    ui3arr['x'] = arr.astype(np.uint32)[:,0]
    ui3arr['y'] = arr.astype(np.uint32)[:,1]
    ui3arr['z'] = arr.astype(np.uint32)[:,2]


# ==========================================
# runtime utilities

def chunk_iterator(nelements, nthreads_per_block=64, max_blocks=1024):
    """Iterator that yields tuples with the values requried to process
    a long array in multiple kernel passes on the GPU.

    Each yielded value is of the form,
        (first_index, elements_this_iteration, nblocks_this_iteration)

    Example:
        >>> list(chunk_iterator(300, 32, 2))
        [(0, 64, 2), (64, 64, 2), (128, 64, 2), (192, 64, 2), (256, 9, 1)]
    """
    first = 0
    while first < nelements:
        elements_left = nelements - first
        blocks = int(elements_left // nthreads_per_block)
        if elements_left % nthreads_per_block != 0:
            blocks += 1 # Round up only if needed
        blocks = min(max_blocks, blocks)
        elements_this_round = min(elements_left, blocks * nthreads_per_block)

        yield (first, elements_this_round, blocks)
        first += elements_this_round

# ==========================================
# runtime utilities

vec_dtypes = set([ x for x in ga.vec.__dict__.values() if type(x) == np.dtype ])

def make_gpu_struct(size, members):
    struct = cuda.mem_alloc(size)

    i = 0
    for n,member in enumerate(members):
        if isinstance(member, ga.GPUArray):
            member = member.gpudata

        if isinstance(member, cuda.DeviceAllocation):
            if i % 8:
                raise Exception('cannot align 64-bit pointer. '
                                'arrange struct member variables in order of '
                                'decreasing size.')

            cuda.memcpy_htod(int(struct)+i, np.getbuffer(np.intp(member)) )
            i += 8
        elif np.isscalar(member) or (hasattr(member, 'dtype') and member.dtype in vec_dtypes and member.shape == ()):
            cuda.memcpy_htod(int(struct)+i, np.getbuffer(member))
            i += member.nbytes
        else:
            raise TypeError('expected a GPU device pointer or scalar type. Instead got %s for member #%d'%(type(member),n))

    return struct

def format_size(size):
    if size < 1e3:
        return '%.1f%s' % (size, ' ')
    elif size < 1e6:
        return '%.1f%s' % (size/1e3, 'K')
    elif size < 1e9:
        return '%.1f%s' % (size/1e6, 'M')
    else:
        return '%.1f%s' % (size/1e9, 'G')

def format_array(name, array):
    return '%-15s %6s %6s' % \
        (name, format_size(len(array)), format_size(array.nbytes))

# ====================================================
# mapped memory tools. might not have opencl analogue

def Mapped(array):
    '''Analog to pycuda.driver.InOut(), but indicates this array
    is memory mapped to the device space and should not be copied.

    To simplify coding, Mapped() will pass anything with a gpudata
    member, like a gpuarray, through unchanged.
    '''
    if gpuapi.is_gpu_api_opencl():
        raise RuntimeError('Command only works for CUDA api')
    return cutools.Mapped(array)

def mapped_alloc(pagelocked_alloc_func, shape, dtype, write_combined):
    '''Returns a pagelocked host array mapped into the CUDA device
    address space, with a gpudata field set so it just works with CUDA 
    functions.'''
    if gpuapi.is_gpu_api_opencl():
        raise RuntimeError('Command only works for CUDA api')
    return cutools.mapped_alloc( pagelocked_alloc_func, shape, dtype, write_combined )

def mapped_empty(shape, dtype, write_combined=False, clcontext=None):
    '''See mapped_alloc()'''
    if gpuapi.is_gpu_api_opencl():
        raise RuntimeError('Command only works for CUDA api')
    return cutools.mapped_empty(shape, dtype, write_combined)

def mapped_empty_like(other, write_combined=False):
    '''See mapped_alloc()'''
    if gpuapi.is_gpu_api_opencl():
        raise RuntimeError('Command only works for CUDA api')
    return cutools.mapped_empty_like(other,write_combined)
                                
def mapped_zeros(shape, dtype, write_combined=False):
    '''See mapped_alloc()'''
    if gpuapi.is_gpu_api_opencl():
        raise RuntimeError('Command only works for CUDA api')    
    return cutools.mapped_zeros(shape,dtype,write_combined)

def mapped_zeros_like(other, write_combined=False):
    '''See mapped_alloc()'''
    if gpuapi.is_gpu_api_opencl():
        raise RuntimeError('Command only works for CUDA api')
    return cutools.mapped_zeros_like(other, write_combined)
                        
