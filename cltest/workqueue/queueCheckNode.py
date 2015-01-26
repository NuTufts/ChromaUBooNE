import chroma.api as api
#from chroma.workqueue.workQueue import workQueue
from workqueue.workQueue import workQueue
from chroma.gpu.gpufuncs import GPUFuncs
from chroma.gpu.tools import get_module, api_options, chunk_iterator, to_float3, copy_to_float3
import pyopencl.array as ga
import pyopencl as cl
import numpy as np

class queueCheckNode(workQueue):

    def __init__(self, context, assigned_workid):
        super( queueCheckNode, self ).__init__( context )
        self.workid = assigned_workid
        # Make work queue
        

    def launchOnce(self, photons, sim, workgroupsize=32):
        # command queue
        comqueue = cl.CommandQueue( self.context )
        if workgroupsize%32!=0:
            raise ValueError('work group size must be multiple value of 32')
        if workgroupsize>self.work_item_sizes:
            raise ValueError('work group size must be smaller than %d'%(self.work_item_sizes))
        
        # photons is instance of GPUPhotons class. contains photon info on the host side
        bvh = sim.detector.bvh

        # get the photons we need to work on
        ourphotons = np.argwhere( photons.requested_workcode==self.workid ) # get index of photons that require us to work on them
        if len(ourphotons)==0:
            return

        # get information on what they need:
        # node list
        max_shared_nodes = self.shared_mem_size/(4*4) # 4 is size of uint32, easy node has 4 of them
        if bvh.nodes.nbytes<self.shared_mem_size:
            # lucky us, we can push the entire node list onto the device (though rarely will be the case)
            node_chunks = [0,len(bvh.nodes)-1]

        nodes = np.take( photons.current_node_index, ourphotons.ravel() ) # node indices

        # planning goals. forget about shared memory for now
        # pack in as many nodes to shared memory as possible.
        # try to take current layer, daughter layer, parent layer in that order
        
        # prep for kernel call
        if api.is_gpu_api_cuda():
            module = get_module( 'wq_checknode.cu', options=api_options, include_source_directory=True)
        elif api.is_gpu_api_opencl():
            module = get_module( 'wq_checknode.cl', self.context, options=api_options, include_source_directory=True)
        gpu_funcs = GPUFuncs(module)

        # gather variables for kernel call
        gpugeo = sim.gpu_geometry
        photon_pos = photons.pos
        photon_dir = photons.dir
        photon_current_node = photons.current_node_index
        photon_tested_node  = ga.to_device( comqueue, np.ones( len(photons.pos), dtype=np.uint32 ) )
        nodes = gpugeo.nodes
        node_parent = ga.to_device( comqueue, sim.detector.node_dsar_tree.parent )
        node_first_daughter = ga.to_device( comqueue, sim.detector.node_dsar_tree.first_daughter )
        node_sibling = ga.to_device( comqueue, sim.detector.node_dsar_tree.sibling )
        node_aunt = ga.to_device( comqueue, sim.detector.node_dsar_tree.aunt )
        world_origin = gpugeo.world_origin
        world_scale  = gpugeo.world_scale
        # make queue related variables
        queue_size = np.int32( len(photons.pos)*2 )
        queue_photon_index = ga.empty( comqueue, queue_size, dtype=np.int32 )
        queue_slot_flag    = ga.zeros( comqueue, queue_size, dtype=np.int32 )
        queue_photon_index[0:len(photons.pos)] = np.arange(0,len(photons.pos), dtype=np.int32)[:]
        queue_photon_index[len(photons.pos):]  = (np.ones( len(photons.pos), dtype=np.int32 )*-1)[:]
        queue_slot_flag[0:len(photons.pos)]    = np.ones( len(photons.pos), dtype=np.int32 )[:]
        a = ga.zeros( comqueue, 1, dtype=ga.vec.uint4 )
        b = np.array( 1, dtype=np.int32 )
        workgroup_photons      = cl.LocalMemory( b.nbytes*workgroupsize )
        workgroup_current_node = cl.LocalMemory( b.nbytes*workgroupsize )
        workgroup_tested_node  = cl.LocalMemory( b.nbytes*workgroupsize )

        max_nodes_can_store = (max_shared_nodes - 10 - 3*workgroupsize )
        max_nodes_can_store -= max_nodes_can_store%32
        max_nodes_can_store = np.int32( max_nodes_can_store )
        loaded_node_start_index = np.int32(0)
        loaded_node_end_index   = np.int32(1)
        workgroup_nodes = cl.LocalMemory( a.nbytes*(max_nodes_can_store+1) )
        max_loops = 4

        if len(gpugeo.extra_nodes)>1:
            raise RuntimeError('did not plan for there to be a node split.')


        print photon_current_node
        print photon_tested_node
        print queue_photon_index
        print queue_slot_flag

        print "Starting node range: ",loaded_node_start_index," to ",loaded_node_end_index
        print "Max nodes in shared: ",max_nodes_can_store
        print "Work group nodes size: ",a.nbytes*workgroupsize," bytes = (",a.nbytes,"*",workgroupsize,")"
        print "Available local memsize: ",self.shared_mem_size
        print "Total number of nodes: ",len(nodes)," (",nodes.nbytes," bytes)"
        print "Stored node size: ",max_nodes_can_store*a.nbytes
        print "Left over: ",self.shared_mem_size-max_nodes_can_store*a.nbytes-a.nbytes*workgroupsize
        print sim.detector.bvh.layer_bounds

        print photon_current_node

        gpu_funcs.checknode( comqueue, (workgroupsize,1,1), (workgroupsize,1,1),
                             np.int32(max_loops),
                             photon_pos.data, photon_dir.data, photon_current_node.data, photon_tested_node.data,
                             nodes.data, node_parent.data, node_first_daughter.data, node_sibling.data, node_aunt.data,
                             world_origin, world_scale, 
                             queue_size, queue_photon_index.data, queue_slot_flag.data, 
                             np.int32(workgroupsize), workgroup_photons, workgroup_current_node, workgroup_tested_node,
                             max_nodes_can_store, workgroup_nodes, loaded_node_start_index, loaded_node_end_index ).wait()

        print photon_current_node.get()
        #print queue_slot_flag.get()
        print len( photon_current_node.get() )
        pass


if __name__ == "__main__":
    # Testing.
    import os,sys
    import chroma.gpu.tools as tools
    os.environ['PYOPENCL_CTX']='0:1'
    context = tools.get_context()
    w = queueCheckNode(context, None)
    w.print_dev_info()
    a = np.array([10],dtype=np.uint32)
    print "max nodes on work item: ",w.shared_mem_size/(4*a.nbytes)
