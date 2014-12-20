#!/usr/bin/env python
import os
import numpy as np
np.seterr(divide='ignore')
from pycuda import autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray

float3 = gpuarray.vec.float3
float4 = gpuarray.vec.float4

print 'device %s' % autoinit.device.name()

current_directory = os.path.split(os.path.realpath(__file__))[0]
from chroma.cuda import srcdir as source_directory

def test_matrix():
    check_matrix((3,3))

def test_matrixfour():
    check_matrix((4,4))

# FIXME: Need to refactor this into proper unit tests
def check_matrix(matrix_shape):
    if matrix_shape == (3,3):
        matrix_type = "matrix"
    elif matrix_shape == (4,4):
        matrix_type = "matrixfour"
    else:
        assert 0
    pass
    matrix_size = matrix_shape[0]*matrix_shape[1]
    path = current_directory + '/%s_test.cu' % matrix_type 
    print "check_matrix %s %s %s " % ( repr(matrix_shape), matrix_type, path ) 

    source = open(path).read()
    mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)
    det = mod.get_function('det')
    inv = mod.get_function('inv')
    matrixadd = mod.get_function('matrixadd')
    matrixsub = mod.get_function('matrixsub')
    matrixmul = mod.get_function('matrixmul')
    multiply = mod.get_function('multiply')
    matrixaddfloat = mod.get_function('matrixaddfloat')
    matrixsubfloat = mod.get_function('matrixsubfloat')
    matrixmulfloat = mod.get_function('matrixmulfloat')
    matrixdivfloat = mod.get_function('matrixdivfloat')
    floataddmatrix = mod.get_function('floataddmatrix')
    floatsubmatrix = mod.get_function('floatsubmatrix')
    floatmulmatrix = mod.get_function('floatmulmatrix')
    floatdivmatrix = mod.get_function('floatdivmatrix')
    matrixaddequals = mod.get_function('matrixaddequals')
    matrixsubequals = mod.get_function('matrixsubequals')
    matrixaddequalsfloat = mod.get_function('matrixaddequalsfloat')
    matrixsubequalsfloat = mod.get_function('matrixsubequalsfloat')
    matrixmulequalsfloat = mod.get_function('matrixmulequalsfloat')
    matrixdivequalsfloat = mod.get_function('matrixdivequalsfloat')
    outer = mod.get_function('outer')
    minusmatrix = mod.get_function('minusmatrix')

    size = {'block': (1,1,1), 'grid': (1,1)}


    for i in range(1):
        a = np.random.random_sample(size=matrix_size).astype(np.float32)
        b = np.random.random_sample(size=matrix_size).astype(np.float32)
        dest = np.empty(1, dtype=np.float32)
        c = np.int32(np.random.random_sample())

        print 'testing det...',

        det(cuda.In(a), cuda.Out(dest), **size)

        if not np.allclose(np.float32(np.linalg.det(a.reshape(*matrix_shape))), dest[0]):
            print 'fail'
            print np.float32(np.linalg.det(a.reshape(*matrix_shape)))
            print dest[0]
        else:
            print 'success'

        print 'testing inv...',

        dest = np.empty(matrix_size, dtype=np.float32)

        inv(cuda.In(a), cuda.Out(dest), **size)

        if not np.allclose(np.linalg.inv(a.reshape(*matrix_shape)).flatten().astype(np.float32), dest):
            print 'fail'
            print np.linalg.inv(a.reshape(*matrix_shape)).flatten().astype(np.float32)
            print dest
        else:
            print 'success'

        print 'testing matrixadd...',

        matrixadd(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)

        if not np.allclose(a+b, dest):
            print 'fail'
            print a+b
            print dest
        else:
            print 'success'

        print 'testing matrixsub...',

        matrixsub(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)

        if not np.allclose(a-b, dest):
            print 'fail'
            print a-b
            print dest
        else:
            print 'sucess'

        print 'testing matrixmul...',

        matrixmul(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)

        if not np.allclose(np.dot(a.reshape(*matrix_shape),b.reshape(*matrix_shape)).flatten(), dest):
            print 'fail'
            print np.dot(a.reshape(*matrix_shape),b.reshape(*matrix_shape)).flatten()
            print dest
        else:
            print 'success'

        print 'testing multiply...',

        x_cpu = np.random.random_sample(size=matrix_shape[0]).astype(np.float32)
        if matrix_shape[0] == 3:
            x_gpu = np.array((x_cpu[0], x_cpu[1], x_cpu[2]), dtype=float3)
            dest = np.empty(1, dtype=float3)
        elif matrix_shape[0] == 4:
            x_gpu = np.array((x_cpu[0], x_cpu[1], x_cpu[2], x_cpu[3]), dtype=float4)
            dest = np.empty(1, dtype=float4)
        else:
            assert 0 
        
        multiply(cuda.In(a), cuda.In(x_gpu), cuda.Out(dest), **size)

        m = a.reshape(*matrix_shape)

        if matrix_shape[0] == 3:
            if not np.allclose(np.dot(x_cpu,m[0]), dest[0]['x']) or \
               not np.allclose(np.dot(x_cpu,m[1]), dest[0]['y']) or \
               not np.allclose(np.dot(x_cpu,m[2]), dest[0]['z']):
                print 'fail'
                print np.dot(x_cpu,m[0])
                print np.dot(x_cpu,m[1])
                print np.dot(x_cpu,m[2])
                print dest[0]['x']
                print dest[0]['y']
                print dest[0]['z']
            else:
                print 'success'
        elif matrix_shape[0] == 4:
            if not np.allclose(np.dot(x_cpu,m[0]), dest[0]['x']) or \
               not np.allclose(np.dot(x_cpu,m[1]), dest[0]['y']) or \
               not np.allclose(np.dot(x_cpu,m[2]), dest[0]['z']) or \
               not np.allclose(np.dot(x_cpu,m[3]), dest[0]['w']):
                print 'fail'
                print np.dot(x_cpu,m[0])
                print np.dot(x_cpu,m[1])
                print np.dot(x_cpu,m[2])
                print np.dot(x_cpu,m[3])
                print dest[0]['x']
                print dest[0]['y']
                print dest[0]['z']
                print dest[0]['w']
            else:
                print 'success'
        else:
            assert 0
 
        print 'testing matrixaddfloat...',

        dest = np.empty(matrix_size, dtype=np.float32)

        matrixaddfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a+c, dest):
            print 'fail'
            print a+c
            print dest
        else:
            print 'success'

        print 'testing matrixsubfloat...',

        matrixsubfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a-c, dest):
            print 'fail'
            print a-c
            print dest
        else:
            print 'success'

        print 'testing matrixmulfloat...',

        matrixmulfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a*c, dest):
            print 'fail'
            print a-c
            print dest
        else:
            print 'success'

        print 'testing matrixdivfloat...',

        matrixdivfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a/c, dest):
            print 'fail'
            print a/c
            print dest
        else:
            print 'success'

        print 'testing floataddmatrix...',

        floataddmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c+a, dest):
            print 'fail'
            print c+a
            print dest
        else:
            print 'success'

        print 'testing floatsubmatrix...',

        floatsubmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c-a, dest):
            print 'fail'
            print c-a
            print dest
        else:
            print 'success'

        print 'testing floatmulmatrix...',

        floatmulmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c*a, dest):
            print 'fail'
            print c*a
            print dest
        else:
            print 'success'

        print 'testing floatdivmatrix...',

        floatdivmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c/a, dest):
            print 'fail'
            print c/a
            print dest
        else:
            print 'success'

        print 'testing matrixaddequals...',

        dest = np.copy(a)

        matrixaddequals(cuda.InOut(dest), cuda.In(b), **size)

        if not np.allclose(a+b, dest):
            print 'fail'
            print a+b
            print dest
        else:
            print 'success'

        print 'testing matrixsubequals...',

        dest = np.copy(a)

        matrixsubequals(cuda.InOut(dest), cuda.In(b), **size)

        if not np.allclose(a-b, dest):
            print 'fail'
            print a-b
            print dest
        else:
            print 'success'

        print 'testing matrixaddequalsfloat...',

        dest = np.copy(a)

        matrixaddequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a+c, dest):
            print 'fail'
            print a+c
            print dest
        else:
            print 'success'

        print 'testing matrixsubequalsfloat...',

        dest = np.copy(a)

        matrixsubequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a-c, dest):
            print 'fail'
            print a-c
            print dest
        else:
            print 'success'

        print 'testing matrixmulequalsfloat...',

        dest = np.copy(a)

        matrixmulequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a*c, dest):
            print 'fail'
            print a*c
            print dest
        else:
            print 'success'

        print 'testing matrixdivequalsfloat...',

        dest = np.copy(a)

        matrixdivequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a/c, dest):
            print 'fail'
            print a*c
            print dest
        else:
            print 'success'

        print 'testing outer...',

        x1_cpu = np.random.random_sample(size=matrix_shape[0]).astype(np.float32)
        x2_cpu = np.random.random_sample(size=matrix_shape[0]).astype(np.float32)

        if matrix_shape[0] == 3:
            x1_gpu = np.array((x1_cpu[0],x1_cpu[1],x1_cpu[2]), dtype=float3)
            x2_gpu = np.array((x2_cpu[0],x2_cpu[1],x2_cpu[2]), dtype=float3)
        elif matrix_shape[0] == 4:
            x1_gpu = np.array((x1_cpu[0],x1_cpu[1],x1_cpu[2],x1_cpu[3]), dtype=float4)
            x2_gpu = np.array((x2_cpu[0],x2_cpu[1],x2_cpu[2],x2_cpu[3]), dtype=float4)
        else:
            assert 0

        outer(x1_gpu, x2_gpu, cuda.Out(dest), **size)

        if not np.allclose(np.outer(x1_cpu, x2_cpu).flatten(), dest):
            print 'fail'
            print np.outer(x1_cpu, x2_cpu).flatten()
            print dest
        else:
            print 'success'

        print 'testing minus matrix...',

        dest = np.copy(a)

        minusmatrix(cuda.In(a), cuda.Out(dest), **size)

        if not np.allclose(-a, dest):
            print 'fail'
            print -a
            print dest
        else:
            print 'success'


if __name__ == '__main__':
    test_matrix()
    test_matrixfour()

