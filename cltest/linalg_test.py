import os, sys
os.environ["PYOPENCL_CTX"] ='1'

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import chroma.gpu.cltools as cltools

float3 = clarray.vec.float3
print "float3 type: ",float3
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
dev = ctx.get_info( cl.context_info.DEVICES )[0]
print 'device %s' % dev.get_info( cl.device_info.NAME )

mod = cltools.get_cl_module( 'linalg_test.cl', ctx, include_source_directory=False )

size = {'block': (256,), 'grid': (1,)}
a_np = np.zeros((size['block'][0],3), dtype=np.float32)
b_np = np.zeros((size['block'][0],3), dtype=np.float32)
c_np = np.float32(np.random.random_sample())
mf = cl.mem_flags

a_vec_np = np.zeros(size['block'][0], dtype=float3)
b_vec_np = np.zeros(size['block'][0], dtype=float3)
d_vec_np = np.zeros(size['block'][0], dtype=float3)
#c_vec_np = np.float32(np.random.random_sample())

#float3add = mod.get_function('float3add')
#float3addequal = mod.get_function('float3addequal')
#float3sub = mod.get_function('float3sub')
#float3subequal = mod.get_function('float3subequal')
#float3addfloat = mod.get_function('float3addfloat')
#float3addfloatequal = mod.get_function('float3addfloatequal')
#floataddfloat3 = mod.get_function('floataddfloat3')
#float3subfloat = mod.get_function('float3subfloat')
#float3subfloatequal = mod.get_function('float3subfloatequal')
#floatsubfloat3 = mod.get_function('floatsubfloat3')
#float3mulfloat = mod.get_function('float3mulfloat')
#float3mulfloatequal = mod.get_function('float3mulfloatequal')
#floatmulfloat3 = mod.get_function('floatmulfloat3')
#float3divfloat = mod.get_function('float3divfloat')
#float3divfloatequal = mod.get_function('float3divfloatequal')
#floatdivfloat3 = mod.get_function('floatdivfloat3')
#dot = mod.get_function('dot')
#cross = mod.get_function('cross')
#norm = mod.get_function('norm')
#minusfloat3 = mod.get_function('minusfloat3')

#a = np.empty(size['block'][0], dtype=float3)
#b = np.empty(size['block'][0], dtype=float3)
#c = np.float32(np.random.random_sample())

a_np[:,0] = np.random.random_sample(size=a_np.shape[0])
a_np[:,1] = np.random.random_sample(size=a_np.shape[0])
a_np[:,2] = np.random.random_sample(size=a_np.shape[0])
a_vec_np['x'] = np.random.random_sample(size=a_vec_np.size)
a_vec_np['y'] = np.random.random_sample(size=a_vec_np.size)
a_vec_np['z'] = np.random.random_sample(size=a_vec_np.size)

b_np[:,0] = np.random.random_sample(size=b_np.shape[0])
b_np[:,1] = np.random.random_sample(size=b_np.shape[0])
b_np[:,2] = np.random.random_sample(size=b_np.shape[0])
b_vec_np['x'] = np.random.random_sample(size=b_vec_np.size)
b_vec_np['y'] = np.random.random_sample(size=b_vec_np.size)
b_vec_np['z'] = np.random.random_sample(size=b_vec_np.size)

#print a_np[0:5]
#print b_np[0:5]
#print c_np
print a_vec_np[0:5]
print b_vec_np[0:5]
print a_vec_np.shape

def testfloat3add():
    dest_np = np.zeros(a_np.shape, dtype=np.float32)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    buf_g = cl.Buffer( ctx, mf.WRITE_ONLY, size=dest_np.nbytes )
    mod.float3add(queue, size["block"], size["grid"], a_g, b_g, buf_g)
    cl.enqueue_copy( queue, dest_np, buf_g )
    print dest_np[0:5,]
    if not np.allclose(a['x']+b['x'], dest['x']) or \
            not np.allclose(a['y']+b['y'], dest['y']) or \
            not np.allclose(a['z']+b['z'], dest['z']):
        assert False
    #if not np.allclose(a_np+b_np, dest_np):
        assert False
    else:
        print "testfloat3add passed."

def testfloat3add_vec():
    dest_np = np.zeros(a_vec_np.shape, dtype=float3)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_vec_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_vec_np)
    buf_g = cl.Buffer( ctx, mf.WRITE_ONLY, size=dest_np.nbytes )
    mod.float3add(queue, size["block"], size["grid"], a_g, b_g, buf_g)
    cl.enqueue_copy( queue, dest_np, buf_g )
    print a_vec_np.dtype, len(a_vec_np.dtype)
    for i in ['x','y','z']:
        d_vec_np[i] = a_vec_np[i]+b_vec_np[i]
    print dest_np[0:5]
    print d_vec_np[0:5]
    #if not np.allclose(a_vec_np['x']+b_vec_np['x'], dest_np['x']) or \
    #        not np.allclose(a_vec_np['y']+b_vec_np['y'], dest_np['y']) or \
    #        not np.allclose(a_vec_np['z']+b_vec_np['z'], dest_np['z']):
    #    assert False
    if not np.allclose( d_vec_np['x'], dest_np['x'] ):
        assert False
    else:
        print "testfloat3add_vec() passed."

def testfloat3sub():
    dest = np.empty(a.size, dtype=float3)
    float3sub(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    if not np.allclose(a['x']-b['x'], dest['x']) or \
            not np.allclose(a['y']-b['y'], dest['y']) or \
            not np.allclose(a['z']-b['z'], dest['z']):
        assert False

def testfloat3addequal():
    dest = np.copy(a)
    float3addequal(cuda.InOut(dest), cuda.In(b), **size)
    if not np.allclose(a['x']+b['x'], dest['x']) or \
            not np.allclose(a['y']+b['y'], dest['y']) or \
            not np.allclose(a['z']+b['z'], dest['z']):
        assert False

def testfloat3subequal():
    dest = np.copy(a)
    float3subequal(cuda.InOut(dest), cuda.In(b), **size)
    if not np.allclose(a['x']-b['x'], dest['x']) or \
            not np.allclose(a['y']-b['y'], dest['y']) or \
            not np.allclose(a['z']-b['z'], dest['z']):
        assert False

def testfloat3addfloat():
    dest = np.empty(a.size, dtype=float3)
    float3addfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']+c, dest['x']) or \
            not np.allclose(a['y']+c, dest['y']) or \
            not np.allclose(a['z']+c, dest['z']):
        assert False

def testfloat3addfloatequal():
    dest = np.copy(a)
    float3addfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']+c, dest['x']) or \
            not np.allclose(a['y']+c, dest['y']) or \
            not np.allclose(a['z']+c, dest['z']):
        assert False

def testfloataddfloat3():
    dest = np.empty(a.size, dtype=float3)
    floataddfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c+a['x'], dest['x']) or \
            not np.allclose(c+a['y'], dest['y']) or \
            not np.allclose(c+a['z'], dest['z']):
        assert False

def testfloat3subfloat():
    dest = np.empty(a.size, dtype=float3)
    float3subfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']-c, dest['x']) or \
            not np.allclose(a['y']-c, dest['y']) or \
            not np.allclose(a['z']-c, dest['z']):
        assert False

def testfloat3subfloatequal():
    dest = np.copy(a)
    float3subfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']-c, dest['x']) or \
            not np.allclose(a['y']-c, dest['y']) or \
            not np.allclose(a['z']-c, dest['z']):
        assert False

def testfloatsubfloat3():
    dest = np.empty(a.size, dtype=float3)
    floatsubfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c-a['x'], dest['x']) or \
            not np.allclose(c-a['y'], dest['y']) or \
            not np.allclose(c-a['z'], dest['z']):
        assert False

def testfloat3mulfloat():
    dest = np.empty(a.size, dtype=float3)
    float3mulfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']*c, dest['x']) or \
            not np.allclose(a['y']*c, dest['y']) or \
            not np.allclose(a['z']*c, dest['z']):
        assert False

def testfloat3mulfloatequal():
    dest = np.copy(a)
    float3mulfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']*c, dest['x']) or \
            not np.allclose(a['y']*c, dest['y']) or \
            not np.allclose(a['z']*c, dest['z']):
        assert False

def testfloatmulfloat3():
    dest = np.empty(a.size, dtype=float3)
    floatmulfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c*a['x'], dest['x']) or \
            not np.allclose(c*a['y'], dest['y']) or \
            not np.allclose(c*a['z'], dest['z']):
        assert False

def testfloat3divfloat():
    dest = np.empty(a.size, dtype=float3)
    float3divfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']/c, dest['x']) or \
            not np.allclose(a['y']/c, dest['y']) or \
            not np.allclose(a['z']/c, dest['z']):
        assert False

def testfloat3divfloatequal():
    dest = np.copy(a)
    float3divfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']/c, dest['x']) or \
            not np.allclose(a['y']/c, dest['y']) or \
            not np.allclose(a['z']/c, dest['z']):
        assert False

def testfloatdivfloat3():
    dest = np.empty(a.size, dtype=float3)
    floatdivfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c/a['x'], dest['x']) or \
            not np.allclose(c/a['y'], dest['y']) or \
            not np.allclose(c/a['z'], dest['z']):
        assert false

def testdot():
    dest = np.empty(a.size, dtype=np.float32)
    dot(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    if not np.allclose(a['x']*b['x'] + a['y']*b['y'] + a['z']*b['z'], dest):
        assert False

def testcross():
    dest = np.empty(a.size, dtype=float3)
    cross(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    for u, v, wdest in zip(a,b,dest):
        w = np.cross((u['x'], u['y'], u['z']),(v['x'],v['y'],v['z']))
        if not np.allclose(wdest['x'], w[0]) or \
                not np.allclose(wdest['y'], w[1]) or \
                not np.allclose(wdest['z'], w[2]):
            print w
            print wdest
            assert False

def testnorm():
    dest = np.empty(a.size, dtype=np.float32)
    norm(cuda.In(a), cuda.Out(dest), **size)

    for i in range(len(dest)):
        if not np.allclose(np.linalg.norm((a['x'][i],a['y'][i],a['z'][i])), dest[i]):
            assert False

def testminusfloat3():
    dest = np.empty(a.size, dtype=float3)
    minusfloat3(cuda.In(a), cuda.Out(dest), **size)
    if not np.allclose(-a['x'], dest['x']) or \
            not np.allclose(-a['y'], dest['y']) or \
            not np.allclose(-a['z'], dest['z']):
        assert False

if __name__ == "__main__":
    #testfloat3add()
    testfloat3add_vec()
