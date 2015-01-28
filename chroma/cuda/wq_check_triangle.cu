//-*-c++-*-
#include "geometry_types.h"
#include "wq_intersect_triangle.h"

extern "C"
{

__device__
void push_onto_queue( uint photon_index, uint &push_pos, uint& push_lt_pop, int* queue_photon_index, int* queue_slot_flag, const int queue_size ) {
  atomicExch( queue_photon_index+push_pos, photon_index );
  atomicExch( queue_slot_flag+push_pos,  1);
  push_pos += 1;
  if ( push_pos>=(uint)queue_size ) {
    push_pos = 0;
    push_lt_pop = 1; // while push has cycled, but pop has not, set this flag
  }
};

__device__
uint pop_from_queue( uint& pop_pos, uint push_pos, uint& push_lt_pop, int* queue_photon_index, int* queue_slot_flag, const int queue_size ) {
  int photon_index = -1;
  if ( pop_pos<push_pos || push_lt_pop==1 ) { // we check this to make sure w don't pass the push boundary
    atomicExch( queue_slot_flag + pop_pos,  0); // pop this slot
    photon_index = atomicExch( queue_photon_index + pop_pos, -1 ); // queue needs to be filled by cpu before launching kernel
    atomicExch( queue_slot_flag + pop_pos, 0 );
    pop_pos += 1;
    if ( pop_pos>=queue_size ) {
      pop_pos = 0;
      push_lt_pop = 0;
    }
  }
  return photon_index;
};

__global__ 
void check_triangle( const int max_loops, 
		     // Photon info [index by photon id] : global array exposed to all work queues
		     const uint nphotons, float3 *positions, float3 *directions, uint* current_node, uint* test_node, int* last_result, float* mindistance, uint* closest_hit_triangle,
		     // Geometry    [indexed by node id] : global array exposed to all work queues
		     const int nnodes, uint4* nodes, uint* node_parent, uint* node_first_daughter, uint* node_sibling, uint* node_aunt,
		     uint3* triangles, // [triangle indices] contains index to vertices
		     float3* vertices, // [vertex indices]
		     const float3 world_origin, const float world_scale,
		     // work queue  [indexed slot]
		     const int queue_size, int* queue_photon_index, int* queue_slot_flag, const int first_empty_slot,
		     // workgroup variables [indexed by local id]
		     const int workgroup_size, const int max_nodes_can_store, 
		     const int loaded_node_start_index, const int loaded_node_end_index, int* out_wavefront_start, int* out_wavefront_end
		     ) {
  int localid = threadIdx.x;

  __shared__ uint pop_pos;
  __shared__ uint push_pos;
  __shared__ uint push_lt_pop;
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
  uint* workgroup_parent   = &shared_mem_block[7*max_nodes_can_store];
  int* workgroup_photons        = (int*)&shared_mem_block[8*max_nodes_can_store];
  int* workgroup_current_node   = (int*)&shared_mem_block[8*max_nodes_can_store + workgroup_size];
  int* workgroup_tested_node    = (int*)&shared_mem_block[8*max_nodes_can_store + 2*workgroup_size];
  

  // initialize local variables
  if ( localid == 0) {
    pop_pos = 0;
    push_pos = (uint)first_empty_slot;
    push_lt_pop = 0; // this flag tracks if push has passed ring boundary and is less than pop index. flips everytime pop or push marker passes boundary
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
    
    // ---------------------------------------------------------------------------
    // Thread-0 fills work items
    if ( localid==0 ) {
      transfer_nodes = false;
      int nfilled = 0;
      for (int i=0; i<workgroup_size; i++) {
	workgroup_photons[ i ] = pop_from_queue( pop_pos, push_pos, push_lt_pop, queue_photon_index, queue_slot_flag, queue_size );
	if ( workgroup_photons[ i ]>=0 ) {
	  workgroup_current_node[ i ]  = current_node[ workgroup_photons[ i ] ];
	  workgroup_tested_node[ i ]   = test_node[ workgroup_photons[ i ] ];	  
	  nfilled++;
	}
	else {
	  workgroup_current_node[ i ] = -1;
	  workgroup_tested_node[ i ] = -1;
	}
      }
      if ( nfilled==0 ) {
	bail = true; // no photons left.
      }
    }

    __syncthreads();
    if ( bail )
      break;

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

    // ---------------------------------------------------------------------------
    // thread zero, polls range of nodes to load into shared memory
    if ( localid == 0) {

      requestnode_min = workgroup_tested_node[ 0 ];
      for (int i=0; i<workgroup_size; i++) {
	if ( workgroup_photons[ i ]>=0 ) { // we ignore non-filled threads
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
	// thread zero could keep pushing and popping photons until it collects enough work items that are within the node front

	// cave man option
	// pull in what we need, but let other threads go to global memory...	
	transfer_nodes = true;
	nodefront_min_index = requestnode_min;
	nodefront_max_index = requestnode_min+max_nodes_can_store;
      }	  
    } //end of thread-0

    // ---------------------------------------------------------------------------
    // workgroup works together to load nodes into local memory
    int num_blocks = (nodefront_max_index-nodefront_min_index)/workgroup_size;
    if ( (nodefront_max_index-nodefront_min_index)%workgroup_size!=0 )
      num_blocks++;
    
    __syncthreads();

    if ( transfer_nodes ) {
      int local_inode;
      int global_inode;
      for (int iblock=0; iblock<num_blocks; iblock++ ) {
    	local_inode = iblock*workgroup_size + localid;
    	global_inode = nodefront_min_index + iblock*workgroup_size + localid;
    	if ( local_inode < max_nodes_can_store && global_inode<nnodes ) {
    	  workgroup_nodes[ local_inode ] = nodes[ global_inode ];
	  workgroup_daughter[ local_inode ] = node_first_daughter[ global_inode ];
	  workgroup_sibling[ local_inode ]  = node_sibling[ global_inode ];
	  workgroup_aunt[ local_inode ]     = node_aunt[ global_inode ];
	}
      }
    }
    __syncthreads();
    
    // ================================================================================================================================
    // -- Now we finally get to testing intersections --
    int photon_index = workgroup_photons[ localid ];
    int nodefront_max = nodefront_max_index; // move to work item memory
    int nodefront_min = nodefront_min_index; // move to work item memory
    
    if ( photon_index>=0 ) { // we ignore non-filled threads

      // get photon information
      float3 photon_pos          = positions[ photon_index ];  // global access
      float3 photon_dir          = directions[ photon_index ]; // global access
      uint local_test_nodeid     = workgroup_tested_node[ localid ]  - nodefront_min; // zero indexed
      uint local_current_nodeid  = workgroup_current_node[ localid ]  - nodefront_min; // zero indexed, this will be the parent
      
      // packup triangle information
      // reach to global memory (probably tough to keep this coherent or store a reasonble, reuseable set into shared)
      Triangle triangle;
      uint3 vertex_indices = triangles[  workgroup_tested_node[ localid ] ];
      triangle.v0 = vertices[ vertex_indices.x ];
      triangle.v1 = vertices[ vertex_indices.y ];
      triangle.v2 = vertices[ vertex_indices.z ];
      // do intersection ( using work item memory )
      float distance_to_triangle = -1;
      float min_triangle_dist = mindistance[ photon_index ];
      int hits = intersect_triangle( photon_pos, photon_dir, triangle, distance_to_center );
      if ( hits ) {
	if ( distance_to_triangle<min_triangle_dist ) {
	  mindistance[ photon_index ] = min_triangle_dist;
	  closest_hit_triangle[ photon_index ] = workgroup_tested_node[ localid ];
	}
	last_result[ photon_index ] = 10; // hit tested triangle
      }
      else {
	last_result[ photon_index ] = 20; // missed tested triangle
      }
      
      // determine next node to dest
      // all leafs are single children, so move to aunt or cousin
      // we need to do this for every thread.
      // unfortunately, there is an unprediactable amount of upwalking to the next node. note only that, global access!! disaster.
      // this is one potential point of significant thread divergence. hopefully, hiding this in its own queue while other work is being done will help
      // algorithm:
      //  -- check for aunt of triangle, sibling of node.
      //  -- if none, move up to parent and check for sibling
      //  -- if sibling is still none, move to grandparent and check for sibling
      //  -- repeat until root is hit (parent=0)
      bool valid_node_found = false;
      uint current_node = workgroup_current_node[ localid ];
      uint next_node;
      uint parent_node = 1;
      if ( local_current_nodeid<max_nodes_can_store )
	next_node = workgroup_sibling[ local_current_nodeid ];
      else
	next_node = node_sibling[ workgroup_tested_node[ localid ] ];
      if ( next_node!=0 )
	valid_node_found = true;
      
      while ( !valid_node_found && parent_node>0 ) {
	local_current_nodeid = current_node - nodefront_min;
	if ( local_current_nodeid < max_nodes_can_store )
	  parent_node = workgroup_parent[ local_current_nodeid ];
	else
	  parent_node = node_parent[ current_node ];
	if ( parent_node-nodefront_min < max_nodes_can_store )
	  next_node   = workgroup_sibling[ parent_node-nodefront_min ];
	else
	  next_node   = node_sibling[ parent_node ];
	if ( next_node!==0 )
	  valid_node_found = true;
      }

      workgroup_current_node[ localid ] = parent_node;
      workgroup_test_node[ localid ] = next_node;
    }// if valid photon


    __syncthreads();
      
    // Now thread 0 pushes threads to end of queue for next step, if not a leaf node
    if ( localid==0 ) {
      // check each thread. if !leaf push to end of work queue
      for (int i=0; i<workgroup_size; i++) {
	
	if ( workgroup_photons[i]>=0 ) {
	  
	  // leaf check: nchild==0
	  uint nchild = 0;
	  if ( workgroup_current_node[ i ]>=nodefront_min && workgroup_current_node[ i ]<nodefront_max_index )
	    nchild = workgroup_nodes[ workgroup_current_node[ i ]-nodefront_min ].w >> CHILD_BITS; // we can get node info from shared memory
	  else
	    nchild = nodes[ workgroup_current_node[ i ] ].w >> CHILD_BITS; // outside node-front, so have to go to global memory
	  if ( nchild==0 ) {
	    // daughter is a leaf. push photon back onto the queue
	    push_onto_queue( workgroup_photons[ i ], push_pos, push_lt_pop, queue_photon_index, queue_slot_flag, queue_size );
	  }
	  else {
	    if ( workgroup_current_node[ i ]>0 ) // internal node
	      last_result[ workgroup_photons[ i ] ] += 3; // exit queue (would go to checknode)
	    else // root node
	      last_result[ workgroup_photons[ i ] ] += 4; // exit queue (would go to physics)
	  }
	  // push to global: atomic to prevent competition with other compute units
	  atomicExch( current_node + workgroup_photons[ i ], workgroup_current_node[ i ] );
	  atomicExch( test_node + workgroup_photons[ i ],    workgroup_tested_node[ i ] );
	}// for good photons only
      }// for loop over work items

      // push loop forward
      iloop += 1;
    }

    __syncthreads();    
    
    // For debug
    /*       // assume it intersects, update photon queue */
    /*       //current_node[ workgroup_photons[ localid ]  ] = workgroup_tested_node[ localid ]; */
    /*       //current_node[ workgroup_photons[ localid ]  ] = workitem_node.w; */
    /*       //current_node[ workgroup_photons[ localid ]  ] = local_test_nodeid; */
    //      current_node[ workgroup_photons[ localid ]  ] = (uint)intersects;

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
