import os
import pytools
import pyopencl as cl
from chroma.cl import srcdir
import numpy as np

cl_options = ()

# ==========================================
# gpu interface utilities

created_contexts = []

def get_cl_module(name, clcontext, options=None, include_source_directory=True, template_uncomment=None, template_fill=None):
    """Returns a pyopencl.Program object from an openCL source file
    located in the chroma cuda directory at cl/[name].

    When template_vars is provided the source is templates substituted, allowing
    dynamic code changes.

    Using lists rather than dicts for 

    """
    if options is None:
        options = []
    elif isinstance(options, tuple):
        options = list(options)
    else:
        raise TypeError('`options` must be a tuple.')

    if include_source_directory:
        options += ['-I',srcdir]

    if os.path.exists( name ):
        with open(name) as f:
            source = f.read()
    elif os.path.exists( srcdir+"/"+name ):
        with open('%s/%s' % (srcdir, name)) as f:
            source = f.read()
    else:
        raise ValueError('Did not find opencl file, %s'%(name))
        

    if template_uncomment is not None:
        source = template_substitute( source, template_uncomment )

    if template_fill is not None:
        source = template_interpolation( source, template_fill )
        if template_fill[0][0] == 'debug' and template_fill[0][1] == 1:
            print source
            
    #print options
    return cl.Program( clcontext, source ).build(options)

#@pytools.memoize
def get_cl_source(name):
    """Get the source code for a openCL source file located in the chroma cuda
    directory at src/[name]."""
    with open('%s/%s' % (srcdir, name)) as f:
        source = f.read()
    return source

def create_cl_context(device=None, context_flags=None):
    """Initialize and return an OpenCL context on the specified device.
    If device_id is None, the default device is used."""
    global created_contexts
    if device==None:
        ctx = cl.create_some_context()
    else:
        ctx = cl.Context( device, properties=context_flags, dev_type=cl.device_type.GPU )
    print "created opencl context: ",ctx
    if ctx not in created_contexts:
        created_contexts.append(ctx)
    return ctx

def close_cl_context(context):
    global created_contexts
    if context in created_contexts:
        created_contexts.remove( context )
    print "closing cl context: ",context
    del context

def get_last_context():
    global created_contexts
    if len(created_contexts)==0:
        created_contexts.append( create_cl_context() )
    return created_contexts[-1]

# ============================================================================
# Host mapped memory buffers
# Quite a bit different than Page_locked device-mapped memory used in pycuda
def mapped_alloc(shape, dtype, write_combined):
    '''Returns a pagelocked host array mapped into the CUDA device
    address space, with a gpudata field set so it just works with CUDA 
    functions.'''
    array = pagelocked_alloc_func(shape=shape, dtype=dtype, mem_flags=flags)
    return array


def mapped_empty( clcontext, shape, dtype, write_combined=False ):
    '''Does not work!'''
    flags = cl.mem_flags.ALLOC_HOST_PTR
    if write_combined:
        flags |= cl.mem_flags.READ_WRITE
        
    mem_host = np.empty( shape, dtype )
    buf_dev = cl.Buffer( clcontext, flags, mem_host.nbytes )
    queue = cl.CommandQueue(clcontext)
    (mem_host2,event) = cl.enqueue_map_buffer( queue, buf_dev, cl.map_flags.WRITE, 0, shape, dtype )
    print buf_dev
    print type(mem_host2),dir(mem_host2),mem_host2.base
    return buf_dev

# ==========================================
# Random number utilities
import chroma.gpu.clrandstate as clrand
def get_rng_states( size, seed=1, cl_context=None ):
    return clrand.get_rng_states( cl_context, size, seed=seed )

def fill_array( context, rng_states, size ):
    queue = cl.CommandQueue(context)
    out_gpu = cl.array.empty( queue, size, dtype=np.float32 )
    randmod = get_cl_module( "random.cl", context, options=api_options, include_source_directory=True)
    randfuncs = GPUFuncs( randmod )
    nthreads_per_block = 256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(size, nthreads_per_block, max_blocks=1):
        randfuncs.fillArray( queue, (nthreads_per_block,1,1), None,
                             np.uint32(first_index),
                             rng_states.data,
                             out_gpu.data )
    out = out_gpu.get()
    return out

