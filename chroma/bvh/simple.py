import chroma.api as api
from chroma.bvh.bvh import BVH
from chroma.gpu.bvh import create_leaf_nodes, merge_nodes, concatenate_layers, optimize_layer
import numpy as np

def make_simple_bvh(mesh, degree):
    '''Returns a BVH tree created by simple grouping of Morton ordered nodes.
    '''
    world_coords, leaf_nodes, morton_codes = \
        create_leaf_nodes(mesh, round_to_multiple=degree)

    #print "morton codes,  ",type(morton_codes),len(morton_codes)
    #print morton_codes[0:10]
    #print morton_codes[-10:]
    #print "leaf nodes ",type(leaf_nodes),len(leaf_nodes)
    #print leaf_nodes[0:10]
    #print leaf_nodes[-10:]

    # rearrange in morton order. NOTE: morton_codes can be shorter than
    # leaf_nodes if dummy padding nodes were added at the end!
    argsort = morton_codes.argsort()
    leaf_nodes[:len(argsort)] = leaf_nodes[argsort]
    assert len(leaf_nodes) % degree == 0
    #print "leaf nodes sorted",type(leaf_nodes)
    #nodeout = open('leafnodes_cl.txt','w')
    #for node in leaf_nodes:
    #    nodeout.write("%s\n"%(node))
    #nodeout.close()

    #if api.is_gpu_api_opencl():
    #    codeout = open('mortoncodes_cl.txt','w')
    #elif api.is_gpu_api_cuda():
    #    codeout = open('mortoncodes_cu.txt','w')
    #for arg in range(0,len(morton_codes)):
    #    codeout.write("%s\n"%(morton_codes[arg]))
    #codeout.close()

    # Create parent layers
    layers = [leaf_nodes]
    ilayer = 0
    while len(layers[0]) > 1:
        #top = optimize_layer(layers[0])
        top = layers[0]
        parent = merge_nodes(top, degree=degree, max_ratio=2)
        layers = [parent, top] + layers[1:]
        #print "Merge output layer=",ilayer,len(parent)
        #print parent
        ilayer += 1

    #raise RuntimeError('stopping for debug')
    # How many nodes total?
    nodes, layer_bounds = concatenate_layers(layers)
    print "concaenated layers: num of nodes=",len(nodes)
    #print nodes[0:50]
    return BVH(world_coords, nodes, layer_bounds[:-1])


    
