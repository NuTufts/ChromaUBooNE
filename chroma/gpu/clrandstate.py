import chroma.api as api
if not api.is_gpu_api_opencl():
    raise RuntimeError('Loading clrandstate when the API is not OpenCL')
from chroma.gpu.gpufuncs import GPUFuncs
import pyopencl as cl
import pyopencl.clrandom as clrand
import numpy as np

randstate_struct_dict = {}

def get_struct_def( context ):
    global randstate_struct_dict
    if context not in randstate_struct_dict:
        randstate_struct = np.dtype( [("a",np.uint32), ("b",np.uint32), ("c",np.uint32), ("d",np.uint32)] )
	#randstate_struct = np.dtype( [("a",np.int32), ("b",np.int32), ("c",np.int32), ("d",np.int32)] )
        print randstate_struct
        device = context.devices[0]
        randstate_struct, my_struct_c_decl = cl.tools.match_dtype_to_c_struct( device, "clrandState", randstate_struct )
        print "Defined clrandState.clrandState struct"
        print my_struct_c_decl
        randstate_struct = cl.tools.get_or_register_dtype("clrandState", randstate_struct )
        print "registered with pyopencl for context ",context
        randstate_struct_dict[ context ] = randstate_struct
    return randstate_struct_dict[ context ]


def get_rng_states(context, size, seed=1):
    queue = cl.CommandQueue(context)
    np.random.seed( seed )
    a = np.random.randint( 0, high=int(0xFFFFFFFF), size=size ).astype(np.uint32)
    b = np.random.randint( 0, high=int(0xFFFFFFFF), size=size ).astype(np.uint32)
    c = np.random.randint( 0, high=int(0xFFFFFFFF), size=size ).astype(np.uint32)
    d = np.random.randint( 0, high=int(0xFFFFFFFF), size=size ).astype(np.uint32)
    np.random.seed(None)

    rand_struct = np.empty( size, get_struct_def(context) )
    rand_struct['a'] = a
    rand_struct['b'] = b
    rand_struct['c'] = c
    rand_struct['d'] = d
    rng_states = cl.array.to_device( queue, rand_struct )
    return rng_states


