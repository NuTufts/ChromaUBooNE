import chroma.api as api
#from chroma.workqueue.workQueue import workQueue
from workqueue.workQueue import workQueue
from chroma.gpu.gpufuncs import GPUFuncs
from chroma.gpu.tools import get_module, api_options, chunk_iterator, to_float3, copy_to_float3
if api.is_gpu_api_opencl():
    import pyopencl.array as ga
    import pyopencl as cl
elif api.is_gpu_api_cuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray as ga
import numpy as np
import time

class queueCheckNode(workQueue):

    def __init__(self, context, assigned_workid):
        super( queueCheckNode, self ).__init__( context )
        self.workid = assigned_workid
        # Make work queue
        

    def launchOnce(self, photons, sim, workgroupsize=32):
        # command queue
        if api.is_gpu_api_opencl():
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
        max_shared_nodes = self.shared_mem_size/((4+7)*4) # 4 is size of uint32, each node has 4 of them, plus daugher, sibling, aunt
        if bvh.nodes.nbytes<self.shared_mem_size:
            # lucky us, we can push the entire node list onto the device (though rarely will be the case)
            node_chunks = [0,len(bvh.nodes)-1]

        nodes = np.take( photons.current_node_index, ourphotons.ravel() ) # node indices

        # planning goals. forget about shared memory for now
        # pack in as many nodes to shared memory as possible.
        # try to take current layer, daughter layer, parent layer in that order
        
        # prep for kernel call
        if api.is_gpu_api_cuda():
            self._call_cuda_kernel( sim, photons, ourphotons, max_shared_nodes, nodes, workgroupsize )
        elif api.is_gpu_api_opencl():
            self._call_opencl_kernel( sim, photons, ourphotons, max_shared_nodes, nodes, workgroupsize, comqueue )


    def _call_opencl_kernel(self, sim, photons, ourphotons, max_shared_nodes, nodes, workgroupsize, comqueue ):
        module = get_module( 'wq_checknode.cl', self.context, options=api_options, include_source_directory=True)
        gpu_funcs = GPUFuncs(module)

        # gather variables for kernel call
        gpugeo = sim.gpu_geometry
        photon_pos = photons.pos
        photon_dir = photons.dir
        photon_current_node = photons.current_node_index
        photon_tested_node  = ga.to_device( comqueue, 1*np.ones( len(photons.pos), dtype=np.uint32 ) )
        photon_last_result  = ga.to_device( comqueue, -1*np.ones( len(photons.pos), dtype=np.int32 ) )
        nodes = gpugeo.nodes
        node_parent = ga.to_device( comqueue, sim.detector.node_dsar_tree.parent )
        node_first_daughter = ga.to_device( comqueue, sim.detector.node_dsar_tree.first_daughter )
        node_sibling = ga.to_device( comqueue, sim.detector.node_dsar_tree.sibling )
        node_aunt = ga.to_device( comqueue, sim.detector.node_dsar_tree.aunt )
        world_origin = gpugeo.world_origin_gpu
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
        c = np.array( 1, dtype=np.uint32 )
        workgroup_photons      = cl.LocalMemory( b.nbytes*workgroupsize )
        workgroup_current_node = cl.LocalMemory( b.nbytes*workgroupsize )
        workgroup_tested_node  = cl.LocalMemory( b.nbytes*workgroupsize )

        max_nodes_can_store = (max_shared_nodes - 20 - 3*workgroupsize )
        max_nodes_can_store -= max_nodes_can_store%32
        max_nodes_can_store = np.int32( max_nodes_can_store )
        loaded_node_start_index = np.int32(0)
        loaded_node_end_index   = np.int32(1)
        node_front_start = ga.empty( comqueue, 1, dtype=np.int32 )
        node_front_end = ga.empty( comqueue, 1, dtype=np.int32 )
        workgroup_nodes = cl.LocalMemory( a.nbytes*(max_nodes_can_store+1) )
        workgroup_daughter = cl.LocalMemory( c.nbytes*(max_nodes_can_store+1) )
        workgroup_sibling = cl.LocalMemory( c.nbytes*(max_nodes_can_store+1) )
        workgroup_aunt = cl.LocalMemory( c.nbytes*(max_nodes_can_store+1) )
        max_loops = 32

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

        print "PRESUB CURRENT NODES"
        print photon_current_node
        print "PRESUB TESTED NODES"
        print photon_tested_node

        start_queue = time.time()
        gpu_funcs.checknode( comqueue, (workgroupsize,1,1), (workgroupsize,1,1),
                             np.int32(max_loops),
                             photon_pos.data, photon_dir.data, photon_current_node.data, photon_tested_node.data, photon_last_result.data,
                             np.int32(len(nodes)), nodes.data, node_parent.data, node_first_daughter.data, node_sibling.data, node_aunt.data,
                             world_origin.data, world_scale, 
                             queue_size, queue_photon_index.data, queue_slot_flag.data, np.int32(len(photon_pos)),
                             np.int32(workgroupsize), workgroup_photons, workgroup_current_node, workgroup_tested_node,
                             max_nodes_can_store, workgroup_nodes, workgroup_daughter, workgroup_sibling, workgroup_aunt,
                             loaded_node_start_index, loaded_node_end_index, node_front_start.data, node_front_end.data ).wait()
        end_queue = time.time()

        print "CheckNode Queue returns. ",end_queue-start_queue," seconds"
        print "(Current node, To Test, result)"
        node_states = zip( photon_current_node.get(), photon_tested_node.get(), photon_last_result.get() )
        for x in xrange(0,len(node_states), 10):
            y = x+10
            if y>len(node_states):
                y = len(node_states)
            print x,": ",node_states[x:y]

        print "LAST RESULT:"
        print photon_last_result.get()

        print "PHOTON QUEUE"
        photon_queue = queue_photon_index.get()
        for x in xrange(0,len(photon_queue), 32):
            y = x+32
            if y>len(photon_queue):
                y = len(photon_queue)
            print x,": ",photon_queue[x:y]

        print "QUEUE SLOT FLAGS"
        slot_flags = queue_slot_flag.get()
        for x in xrange(0,len(slot_flags), 32):
            y = x+32
            if y>len(slot_flags):
                y = len(slot_flags)
            print x,": ",slot_flags[x:y]

        print "NODE FRONT: ",node_front_start.get(), " to ",node_front_end.get(), node_front_end.get()-node_front_start.get()

    def _call_cuda_kernel(self, sim, photons, ourphotons, max_shared_nodes, nodes, workgroupsize ):
        module = get_module( 'wq_checknode.cu', options=api_options, include_source_directory=True)
        gpu_funcs = GPUFuncs(module)

        # gather variables for kernel call
        gpugeo = sim.gpu_geometry
        photon_pos = photons.pos
        photon_dir = photons.dir
        photon_current_node = photons.current_node_index
        photon_tested_node  = ga.to_gpu( 1*np.ones( len(photons.pos), dtype=np.uint32 ) )
        photon_last_result  = ga.to_gpu( -1*np.ones( len(photons.pos), dtype=np.int32 ) )
        nodes = gpugeo.nodes
        node_parent = ga.to_gpu( sim.detector.node_dsar_tree.parent )
        node_first_daughter = ga.to_gpu( sim.detector.node_dsar_tree.first_daughter )
        node_sibling = ga.to_gpu( sim.detector.node_dsar_tree.sibling )
        node_aunt = ga.to_gpu( sim.detector.node_dsar_tree.aunt )
        world_origin = gpugeo.world_origin
        world_scale  = gpugeo.world_scale

        # make queue related variables
        queue_size = np.int32( len(photons.pos)*2 )
        queue_photon_index = ga.empty( queue_size, dtype=np.int32 )
        queue_slot_flag    = ga.zeros( queue_size, dtype=np.int32 )
        queue_photon_index[0:len(photons.pos)].set( np.arange(0,len(photons.pos), dtype=np.int32)[:] )
        queue_photon_index[len(photons.pos):].set( -1*np.ones( len(photons.pos), dtype=np.int32 ) )
        queue_slot_flag[0:len(photons.pos)].set( np.ones( len(photons.pos), dtype=np.int32 )[:] )
        a = ga.zeros( 1, dtype=ga.vec.uint4 )
        b = np.array( 1, dtype=np.int32 )
        c = np.array( 1, dtype=np.uint32 )
        
        max_nodes_can_store = (max_shared_nodes - 20 - 3*workgroupsize )
        max_nodes_can_store -= max_nodes_can_store%32
        max_nodes_can_store = np.int32( max_nodes_can_store )

        loaded_node_start_index = np.int32(0)
        loaded_node_end_index   = np.int32(1)
        node_front_start = ga.empty( 1, dtype=np.int32 )
        node_front_end = ga.empty( 1, dtype=np.int32 )
        
        max_loops = 1000

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

        print "PRESUB CURRENT NODES"
        print photon_current_node
        print "PRESUB TESTED NODES"
        print photon_tested_node
        print "STARTING QUEUE"
        print queue_photon_index

        start_queue = time.time()
        gpu_funcs.checknode( np.int32(max_loops),
                             photon_pos, photon_dir, photon_current_node, photon_tested_node, photon_last_result,
                             np.int32(len(nodes)), nodes, node_parent, node_first_daughter, node_sibling, node_aunt,
                             world_origin, world_scale, 
                             queue_size, queue_photon_index, queue_slot_flag, np.int32(len(photon_pos)),
                             max_nodes_can_store,
                             loaded_node_start_index, loaded_node_end_index, node_front_start, node_front_end,
                             block=(workgroupsize,1,1), grid=(1,1), shared=4*(7*max_nodes_can_store + 3*workgroupsize + 1) )
        cuda.Context.get_current().synchronize()
        end_queue = time.time()

        nactive = len( np.argwhere( queue_slot_flag.get()==1 ) )

        print "CheckNode Queue returns. ",end_queue-start_queue," seconds"
        print "(Current node, To Test)"
        node_states = zip( photon_current_node.get(), photon_tested_node.get(), photon_last_result.get() )
        for x in xrange(0,len(node_states), 10):
            y = x+10
            if y>len(node_states):
                y = len(node_states)
            print x,": ",node_states[x:y]

        print "LAST RESULT:"
        np_photon_results = photon_last_result.get()
        for x in xrange(0,len(np_photon_results), 10):
            y = x+10
            if y>len(np_photon_results):
                y = len(np_photon_results)
            print x,": ",np_photon_results[x:y]

        print "PHOTON QUEUE"
        photon_queue = queue_photon_index.get()
        for x in xrange(0,len(photon_queue), 10):
            y = x+10
            if y>len(photon_queue):
                y = len(photon_queue)
            print x,": ",photon_queue[x:y]

        print "QUEUE SLOT FLAGS: ",nactive," threads"
        slot_flags = queue_slot_flag.get()
        for x in xrange(0,len(slot_flags), 10):
            y = x+10
            if y>len(slot_flags):
                y = len(slot_flags)
            print x,": ",slot_flags[x:y]

        print "NODE FRONT: ",node_front_start.get(), " to ",node_front_end.get(), node_front_end.get()-node_front_start.get()


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
