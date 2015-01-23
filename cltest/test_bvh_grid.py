import os
#os.environ["PYOPENCL_CTX"] ='0:1'
from unittest_find import unittest
import chroma.api as api
api.use_opencl()
import chroma.gpu.tools as tools
import pyopencl as cl
from chroma.bvh import BVH
from chroma.bvh.grid import make_recursive_grid_bvh
from chroma.bvh.bvh import node_areas
import chroma.models

import numpy as np
#from numpy.testing import assert_array_max_ulp, assert_array_equal, \
#    assert_approx_equal

def build_simple_bvh(degree):
    #mesh = chroma.models.lionsolid()
    mesh = chroma.models.companioncube()
    bvh = make_recursive_grid_bvh(mesh, degree,save_morton_codes="mortonout.cl.cpu.laptop.double.txt")

    nodes = bvh.nodes
    layer_bounds = np.append(bvh.layer_offsets, len(nodes))
    world_coords = bvh.world_coords

    out = open('gridout.bvh','w')
    for node in nodes:
        print>>out,node

    for i, (layer_start, layer_end) in enumerate(zip(layer_bounds[:-1], layer_bounds[1:])):
        print i, node_areas(nodes[layer_start:layer_end]).sum() * world_coords.world_scale**2

    return bvh
    #assert isinstance(bvh, BVH)

def test_simple():
    yield build_simple_bvh, 2
    yield build_simple_bvh, 3
    yield build_simple_bvh, 4

if __name__ == "__main__":
    bvh = build_simple_bvh(3)
    print bvh.layer_bounds
    for ilayer in xrange(len(bvh.layer_bounds)):
        print bvh.nodes[ bvh.layer_bounds[ilayer]:bvh.layer_bounds[ilayer+1] ]
    #from chroma.bvh.NodeDSAR import NodeDSARtree
    #tree = NodeDSARtree( bvh )
