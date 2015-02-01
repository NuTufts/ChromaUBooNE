import os,sys
#os.environ['PYOPENCL_CTX']='0:1'
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['CUDA_PROFILE'] = '1'
import chroma.api as api
#api.use_opencl()
api.use_cuda()
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from unittest_find import unittest
import numpy as np
from chroma.sim import Simulation
from chroma.event import Photons
from chroma.uboone.uboonedet import ubooneDet
from chroma.gpu.photon import GPUPhotons
from chroma.gpu.tools import get_module, api_options
from chroma.gpu.gpufuncs import GPUFuncs

try:
    import ROOT as rt
    has_root = True
except:
    has_root = False

daefile = "../uboone/dae/microboone_32pmts_nowires_cryostat.dae"
geo = ubooneDet( daefile, detector_volumes=["vol_PMT_AcrylicPlate","volPaddle_PMT"],
                 acrylic_detect=True, acrylic_wls=False,  
                 read_bvh_cache=True, cache_dir="./uboone_cache",
                 dump_node_info=True)
sim = Simulation(geo, geant4_processes=0)
origin = geo.bvh.world_coords.world_origin

nodes      = sim.gpu_geometry.nodes
extra_node = sim.gpu_geometry.extra_nodes
triangles  = sim.gpu_geometry.triangles
vertices   = sim.gpu_geometry.vertices
print vertices.shape
vertices4  = np.zeros( (len(vertices), 4), dtype=np.float32 )
print vertices.get().ravel().view( np.float32 ).shape
vertices4[:,:-1] = vertices.get().ravel().view( np.float32 ).reshape( len(vertices),3 )

module = get_module('test_texture.cu', options=api_options, include_source_directory=True)
gpu_funcs = GPUFuncs(module)
node_texture_ref       = module.get_texref( "node_tex_ref" )
extra_node_texture_ref = module.get_texref( "extra_node_tex_ref" )
triangles_texture_ref  = module.get_texref( "triangles_tex_ref" )
vertices_texture_ref   = module.get_texref( "vertices_tex_ref" )

node_vec_texture_ref   = module.get_texref( "nodevec_tex_ref" )
node_vec_texture_ref.set_format( cuda.array_format.UNSIGNED_INT32, 4 )

ur_nodes = nodes.get().ravel().view( np.uint32 )
ur_nodes_gpu = ga.to_gpu( ur_nodes )
ur_nodes_gpu.bind_to_texref_ext( node_texture_ref )
nodes_nbytes = ur_nodes.nbytes

ur_nodes = nodes.get().ravel().view( np.uint32 )
ur_nodes_vec_gpu = ga.to_gpu( ur_nodes )
node_vec_texture_ref.set_address( ur_nodes_vec_gpu.gpudata, ur_nodes_vec_gpu.nbytes )

ur_extra_node = extra_node.ravel().view( np.uint32 )
ur_extra_node_gpu = ga.to_gpu( ur_extra_node )
ur_extra_node_gpu.bind_to_texref_ext( extra_node_texture_ref )
extra_nbytes = ur_extra_node.nbytes

ur_triangles = triangles.get().ravel().view( np.uint32 )
ur_triangles_gpu = ga.to_gpu( ur_triangles )
ur_triangles_gpu.bind_to_texref_ext( triangles_texture_ref )
triangles_nbytes = ur_triangles.nbytes

ur_vertices = vertices.get().ravel().view( np.float32 )
ur_vertices_gpu = ga.to_gpu( ur_vertices )
#ur_vertices_gpu.bind_to_texref_ext( vertices_texture_ref )
vertices_nbytes = ur_vertices.nbytes


vertices_vec_texture_ref = module.get_texref( "verticesvec_tex_ref" )
vertices_vec_texture_ref.set_format( cuda.array_format.FLOAT, 4 )
ur_vertices_vec_gpu = ga.to_gpu( vertices4.ravel().view( np.float32 ) )
vertices_vec_texture_ref.set_address( ur_vertices_vec_gpu.gpudata, ur_vertices_vec_gpu.nbytes )
print vertices4[:5]

print "nodes: ",nodes_nbytes/1000.0," kB"
print "extra nodes: ",extra_nbytes/1000.0," kB"
print "vertices: ",vertices_nbytes/1000.0," kB"
print "triangles: ",triangles_nbytes/1000.0," kB"
print "geo: ",(nodes_nbytes+extra_nbytes+vertices_nbytes+triangles_nbytes)/1000.0," kB"
print nodes[0:5]
gpu_funcs.test_texture( np.int32(5), ur_nodes_gpu,
                        block=(64,1,1), grid=(1,1) )

