import os
import pyopencl as cl
from chroma.cl import srcdir

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
        options += ['-I' + srcdir]

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
         

    return cl.Program( clcontext, source ).build(options)

