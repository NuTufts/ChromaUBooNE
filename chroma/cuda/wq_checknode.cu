//-*-c++-*-
#include "geometry_types.h"
#include "wq_intersect_bvh.h"

extern "C"
{

__global__ 
void checknode( const int max_loops, 
		// Photon info [index by photon id]
		float3 *positions, float3 *directions, uint* current_node, uint* test_node, int* last_result,
		// Geometry    [indexed by node id]
		const int nnodes, uint4* nodes, uint* node_parent, uint* node_first_daughter, uint* node_sibling, uint* node_aunt,
		const float3 world_origin, const float world_scale,
		// work queue  [indexed slot]
		const int queue_size, int* queue_photon_index, int* queue_slot_flag, const int first_empty_slot,
		// workgroup variables [indexed by local id]
		const int workgroup_size, const int max_nodes_can_store, 
		const int loaded_node_start_index, const int loaded_node_end_index, int* out_wavefront_start, int* out_wavefront_end
		) {
  int localid = threadIdx.x;
  //int groupid = blockIdx.x;

  __shared__ uint pop_pos;
  __shared__ uint push_pos;
  __shared__ uint nodefront_min_index;
  __shared__ uint nodefront_max_index;
  __shared__ uint requestnode_min;
  __shared__ uint requestnode_max;
  __shared__ bool transfer_nodes;
  __shared__ bool bail;
  __shared__ uint iloop;

  extern __shared__ uint shared_mem_block[];
  uint4* workgroup_nodes   = (uint4*)&shared_mem_block[0];
  uint* workgroup_daughter = &shared_mem_block[4*max_nodes_can_store];
  uint* workgroup_sibling  = &shared_mem_block[5*max_nodes_can_store];
  uint* workgroup_aunt     = &shared_mem_block[6*max_nodes_can_store];
  int* workgroup_photons        = (int*)&shared_mem_block[7*max_nodes_can_store];
  int* workgroup_current_node   = (int*)&shared_mem_block[7*max_nodes_can_store + workgroup_size];
  int* workgroup_tested_node    = (int*)&shared_mem_block[7*max_nodes_can_store + 2*workgroup_size];
  

  // initialize local variables
  if ( localid == 0) {
    pop_pos = 0;
    push_pos = (uint)first_empty_slot;
    nodefront_min_index = loaded_node_start_index;
    nodefront_max_index = loaded_node_end_index;
    requestnode_min = 0;
    requestnode_max = 1;
    iloop = 0;
    bail = false;
  }

  __syncthreads();

  // WORK QUEUE LOOP STARTS HERE
  int thread_iloop = iloop;
  int queue_index;

  while (thread_iloop<max_loops) {

    // warp fill
    // all work items load requested photon and node
    queue_index = pop_pos + localid; // this can overrun!
    if ( queue_index < queue_size && queue_slot_flag[queue_index]==1 ) {
      atomicExch( queue_slot_flag + queue_index,  0); // pop this slot
      workgroup_photons[ localid ]       = queue_photon_index[ queue_index ]; // queue needs to be filled by cpu before launching kernel
      queue_photon_index[ queue_index ] = -1;
      workgroup_current_node[ localid ]  = current_node[ workgroup_photons[ localid ] ];
      workgroup_tested_node[ localid ]   = test_node[ workgroup_photons[ localid ] ];
    }
    else if ( queue_index>=queue_size && queue_slot_flag[queue_index-queue_size]==1) {
      atomicExch( queue_slot_flag + queue_index-queue_size,  0); // pop this slot
      workgroup_photons[ localid ]       = queue_photon_index[ queue_index-queue_size ]; // queue needs to be filled by cpu before launching kernel
      queue_photon_index[ queue_index-queue_size ] = -1;
      workgroup_current_node[ localid ]  = current_node[ workgroup_photons[ localid ] ];
      workgroup_tested_node[ localid ]   = test_node[ workgroup_photons[ localid ] ];
    }
    else {
      workgroup_photons[ localid ] = -1;
      workgroup_current_node[ localid ] = -1;
      workgroup_tested_node[ localid ] = -1;
    }

    __syncthreads();

    // pop last 16
    if ( localid == 0) {
      int next_pos = pop_pos + (uint)workgroup_size;
      transfer_nodes = false;
      // move queue position
      if ( next_pos >= queue_size ) 
	next_pos = next_pos-queue_size;
      pop_pos = next_pos;
    }

    __syncthreads();

    /* // ------------------------------------------------------ */
    /* // for debug (checks pop above) */
    /* current_node[ workgroup_photons[ localid ] ] = localid; */
    /* if ( localid==0 ) { */
    /*   iloop += 1; */
    /*   // push back onto queue */
    /*   for (int i=0; i<workgroup_size; i++) { */
    /* 	queue_photon_index[ push_pos ] = workgroup_photons[ i ]; */
    /* 	atomicExch( queue_slot_flag+push_pos,  1); */
    /* 	push_pos += 1; */
    /* 	if ( push_pos>=(uint)queue_size ) */
    /* 	  push_pos = 0; */
    /*   } */
    /* } */
    /* barrier( CLK_LOCAL_MEM_FENCE ); */
    /* thread_iloop = iloop;     */
    /* continue; */
    /* // ------------------------------------------------------ */

    // thread zero, polls range of nodes to get
    if ( localid == 0) {

      requestnode_min = workgroup_tested_node[ 0 ];
      for (int i=0; i<workgroup_size; i++) {
	if ( workgroup_photons[ i ]>=0 ) { // we ignore non-filled threads
	  //requestnode_min = min( requestnode_min, (uint)workgroup_current_node[ i ] );
	  requestnode_min = min( requestnode_min, (uint)workgroup_tested_node[ i ] );
	  requestnode_max = max( requestnode_max, (uint)workgroup_tested_node[ i ] );
	}
      }

      // easy scenario, we can fit all requested nodes (or more) into shared memory

      if ( (requestnode_max-requestnode_min)<=(uint)max_nodes_can_store ) {
	requestnode_max = requestnode_min+max_nodes_can_store;
	// if we don't have the nodes we need, schedule a transfer, update the front
	if ( requestnode_min<nodefront_min_index || requestnode_max>nodefront_max_index ) {
	  transfer_nodes = true;
	  nodefront_min_index = requestnode_min;
	  nodefront_max_index = requestnode_max;
	}
      }
      else {
	// hard scenario, we can't load them all in. 

	// fancy option
	// thread zero could keep pushing and popping photons until it collects enough work items
	//requestnode_max = requestnode_min+max_nodes_can_store;
	/* for (int i=0; i<workgroup_size; i++) { */
	/*   if ( workgroup_tested_node[ i ]>requestnode_max ) { */
	/*     // push photon item back onto queue */
	/*     queue_photon_index[ push_pos ] = workgroup_photons[ i ]; */
	/*     atomicExch( queue_slot_flag+push_pos,  1); */
	/*     push_pos += 1; */
        /*     if ( push_pos>=(uint)queue_size ) */
        /*       push_pos = 0; */

	/*     // pop end of photon */
	/*     workgroup_photons[ i ] = queue_photon_index[ pop_pos ]; */
	/*     workgroup_current_node[ i ] = current_node[ workgroup_photons[ i ] ]; */
	/*     workgroup_tested_node[ localid ] = test_node[ workgroup_photons[ i ] ]; */
	/*     atomicExch( queue_slot_flag+pop_pos,  0); */
	/*     queue_photon_index[ pop_pos ] = -2; */
	/*     pop_pos += 1; */
	/*     if ( pop_pos>=(uint)queue_size ) */
        /*       pop_pos = 0; */
	/*   } */
	/* } */

	// cave man option
	// pull in what we need, but let other threads go to global memory...	
	transfer_nodes = true;
	nodefront_min_index = requestnode_min;
	nodefront_max_index = requestnode_min+max_nodes_can_store;
      }	  
    } //end of thread-0

    // workgroup works together to load nodes into local memory
    int num_blocks = (nodefront_max_index-nodefront_min_index)/workgroup_size;
    if ( (nodefront_max_index-nodefront_min_index)%workgroup_size!=0 )
      num_blocks++;

    __syncthreads();
    if ( bail )
      break;

    if ( transfer_nodes ) {
      for (int iblock=0; iblock<num_blocks; iblock++ ) {
    	int local_inode = iblock*workgroup_size + localid;
    	int global_inode = nodefront_min_index + iblock*workgroup_size + localid;
    	if ( local_inode < max_nodes_can_store && global_inode<nnodes ) {
    	  workgroup_nodes[ local_inode ] = nodes[ global_inode ];
	  workgroup_daughter[ local_inode ] = node_first_daughter[ global_inode ];
	  workgroup_sibling[ local_inode ]  = node_sibling[ global_inode ];
	  workgroup_aunt[ local_inode ]     = node_aunt[ global_inode ];
	}
      }
    }
    __syncthreads();

    // -- Now we finally get to testing intersections --
    
    if ( workgroup_photons[ localid ]>=0 ) { // we ignore non-filled threads

      // get photon information
      float3 photon_pos = positions[ workgroup_photons[ localid ] ];  // global access
      float3 photon_dir = directions[ workgroup_photons[ localid ] ]; // global access
      uint local_test_nodeid    = workgroup_tested_node[ localid ]  - nodefront_min_index; // zero index
      //uint local_current_nodeid = workgroup_current_node[ localid ] - nodefront_min_index; // zero index

      // get and unpack node
      Node node_struct;
      uint4 workitem_node;
      if ( local_test_nodeid<max_nodes_can_store )
	workitem_node = workgroup_nodes[ local_test_nodeid ]; // get the tested node
      else
	workitem_node = nodes[ workgroup_tested_node[ localid ] ]; // non-localized, warped global access :(
      uint3 lower_int = make_uint3(workitem_node.x & 0xFFFF, workitem_node.y & 0xFFFF, workitem_node.z & 0xFFFF);
      uint3 upper_int = make_uint3(workitem_node.x >> 16, workitem_node.y >> 16, workitem_node.z >> 16);
      float3 flower = make_float3( lower_int.x, lower_int.y, lower_int.z );
      float3 fupper = make_float3( upper_int.x, upper_int.y, upper_int.z );
      node_struct.lower = world_origin + flower * world_scale;
      node_struct.upper = world_origin + fupper * world_scale;
      node_struct.child = workitem_node.w & ~NCHILD_MASK;
      node_struct.nchild = workitem_node.w >> CHILD_BITS;
      
      int intersects = intersect_internal_node( photon_pos, photon_dir, node_struct );
      last_result[ workgroup_photons[ localid ] ] = intersects;
      
      if ( intersects ) {
	// passes. update current node to test node.  set test node as first daughter of new node
	uint next_daughter;
	if ( local_test_nodeid<max_nodes_can_store )
	  next_daughter = workgroup_daughter[ local_test_nodeid ];
	else
	  next_daughter = node_first_daughter[ workgroup_tested_node[ localid ] ];
	// store next nodes in local space first (later we will push the info into global memory
	workgroup_current_node[ localid ] =  workgroup_tested_node[ localid ];
	workgroup_tested_node[ localid ] = next_daughter;
      }
      else {
	// does not pass.  check sibling of tested node.
	uint sibling;
	uint aunt;
	if ( local_test_nodeid<max_nodes_can_store ) {
	  sibling = workgroup_sibling[ local_test_nodeid ];
	  aunt    = workgroup_aunt[ local_test_nodeid ];
	}
	else {
	  sibling = node_sibling[ workgroup_tested_node[ localid ] ];
	  aunt    = node_aunt[ workgroup_tested_node[ localid ] ];
	}
	// current node is unchanged
	workgroup_tested_node[ localid ] = sibling;
	if ( workgroup_tested_node[ localid ]==0 )
	  workgroup_tested_node[ localid ] = aunt;
      }
      
      __syncthreads();
      
      // Now thread 0 pushes threads to end of queue for next step, if not a leaf node
      if ( localid==0 ) {
	// check each thread. if !leaf or !non-aunt, push to end of work queue
	for (int i=0; i<workgroup_size; i++) {
	  uint nchild = 0;
	  if ( workgroup_current_node[ i ]>=nodefront_min_index && workgroup_current_node[ i ]<nodefront_max_index )
	    nchild = workgroup_nodes[ workgroup_current_node[ i ]-nodefront_min_index ].w >> CHILD_BITS;
	  else
	    nchild = nodes[ workgroup_current_node[ i ] ].w >> CHILD_BITS; // outside node-front, so have to go to global memory
	  if ( nchild>0 ) {
	    // internal node
	    queue_photon_index[ push_pos ] = workgroup_photons[ i ];
	    atomicExch( queue_slot_flag+push_pos,  1);
	    push_pos += 1;
	    if ( push_pos>=(uint)queue_size )
	      push_pos = 0;
	  }
	  else {
	    
	    last_result[ workgroup_photons[ i ] ] = 2; //leaf node
	  }
	  // push to global: atomic to prevent competition with other compute units
	  atomicExch( current_node + workgroup_photons[ i ], workgroup_current_node[ i ] );
	  atomicExch( test_node + workgroup_photons[ i ],    workgroup_tested_node[ i ] );

	}
      }
      
      // For debug
/*       // assume it intersects, update photon queue */
/*       //current_node[ workgroup_photons[ localid ]  ] = workgroup_tested_node[ localid ]; */
/*       //current_node[ workgroup_photons[ localid ]  ] = workitem_node.w; */
/*       //current_node[ workgroup_photons[ localid ]  ] = local_test_nodeid; */
//      current_node[ workgroup_photons[ localid ]  ] = (uint)intersects;
    } // if valid photon

    __syncthreads();    

    if ( localid==0 ) {
      iloop += 1;
    }
    __syncthreads();
    thread_iloop = iloop;
    //barrier( CLK_LOCAL_MEM_FENCE );
  } // end of while loop

  if ( localid==0 ) {
    *out_wavefront_start = nodefront_min_index;
    *out_wavefront_end   = nodefront_max_index;
  }

  return;

}

}// end of extern C
