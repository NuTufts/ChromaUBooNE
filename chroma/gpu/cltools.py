import os
import pytools
import pyopencl as cl
from chroma.cl import srcdir

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
            
    print options
    return cl.Program( clcontext, source ).build(options)

@pytools.memoize
def get_cl_source(name):
    """Get the source code for a openCL source file located in the chroma cuda
    directory at src/[name]."""
    with open('%s/%s' % (srcdir, name)) as f:
        source = f.read()
    return source

def create_cl_context(device=None, context_flags=None):
    """Initialize and return an OpenCL context on the specified device.
    If device_id is None, the default device is used."""
    if device==None:
        ctx = cl.create_some_context()
    else:
        ctx = cl.Context( device, properties=context_flags, dev_type=cl.device_type.GPU )
    print "created opencl context: ",ctx
    return ctx

def get_last_context():
    global default_context
    if len(created_contexts)==0:
        created_contexts.append( create_cl_context() )
    return created_contexts[-1]

