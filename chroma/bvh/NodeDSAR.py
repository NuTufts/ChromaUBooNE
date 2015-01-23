from chroma.bvh import unpack_nodes
import numpy as np

# This class stores Node DAR information:
# DSAR = first Daughter, Sibling, Aunt Ropes
# Goal is to use this information to allow for stackless navigation


class NodeDSAR:
    NOPARENT   = np.uint32(0)
    NOAUNT     = np.uint32(0)
    NOSIBLING  = np.uint32(0)
    NODAUGHTER = np.uint32(0)
    def __init__(self, node_index, parent=NOPARENT, fdaughter=NODAUGHTER, sibling=NOSIBLING, aunt=NOAUNT, ndaughters=1 ):
        self.node_index     = node_index
        self.parent         = parent
        self.first_daughter = fdaughter
        self.sibling        = sibling
        self.aunt           = aunt
        self.ndaughters     = ndaughters
        self.leafnode       = False
        self.fragment     = False

    def __str__(self):
        if not self.leafnode and not self.fragment:
            return "NodeDSAR() INTERNAL %u: parent=%u aunt=%d sibling=%d first_daughter=%d ndaughters=%d" % ( self.node_index, 
                                                                                                              self.parent , self.aunt, 
                                                                                                              self.sibling, self.first_daughter, self.ndaughters )
        else:
            if self.leafnode:
                return "NodeDSAR() LEAF %u: parent=%u aunt=%d sibling=%d first_daughter=%d ndaughters=%d" % ( self.node_index, 
                                                                                                              self.parent , self.aunt, 
                                                                                                              self.sibling, self.first_daughter, self.ndaughters )
            elif self.fragment:
                return "NodeDSAR() FRAGMENT %u: parent=%u aunt=%d sibling=%d first_daughter=%d ndaughters=%d" % ( self.node_index, 
                                                                                                                  self.parent , self.aunt, 
                                                                                                                  self.sibling, self.first_daughter, self.ndaughters )

    def isleaf(self):
        self.leafnode = True

    def isfragment(self):
        self.fragment = True


class NodeDSARtree:
    def __init__( self, bvh ):
        self.layer_bounds = bvh.layer_bounds
        self.nodes = bvh.nodes
        self.dsartree = {}

        self._make_tree( self.layer_bounds, self.nodes )
        self._make_arrays()

    def _make_tree( self, layer_bounds, nodes ):
        # We go layer by layer
        # (1) Get daughter nodes
        # (2) Make sure they are sorted by morton number (maybe not critical)
        # (3) assign them their sibling, first daughter and aunt number
        # (4) Go to next layer

        parent_node_dict = None # packed dict[child] = {parent:index, aunt:index )
        for ilayer in range(0,len(layer_bounds)):
            # Get layer nodes
            layer_start = layer_bounds[ilayer]
            if ilayer+1<len(layer_bounds):
                layer_end   = layer_bounds[ilayer+1]
            else:
                layer_end = len(nodes)
            packed_layer_nodes = nodes[layer_start:layer_end]
            layer_nodes = unpack_nodes( packed_layer_nodes )
            #print "========================================"
            #print "LAYER %d: [%d, %d] (Total nodes: %s)"  % ( ilayer, layer_start, layer_end, len(nodes) )
            
            # Loop over nodes
            for inode, node in enumerate(layer_nodes):
                node_index    = np.uint32(layer_start + inode)
                sibling_index = node_index+1
                if sibling_index>=layer_end:
                    sibling_index = NodeDSAR.NOSIBLING
                daughter_index = node['child']
                ndaughters = node['nchild']
                isleaf = False
                isfragment = False
                if ilayer>0:
                    if ndaughters>0:
                        if node_index in parent_node_dict:
                            parent_index = parent_node_dict[node_index]['parent']
                            aunt_index   = parent_node_dict[node_index]['aunt']
                            if sibling_index not in parent_node_dict or parent_node_dict[sibling_index]['parent']!=parent_index:
                                sibling_index = NodeDSAR.NOSIBLING # actually a cousin
                        else:
                            # unsure why these nodes exist. is it a bug?
                            parent_index = NodeDSAR.NOPARENT
                            aunt_index   = NodeDSAR.NOAUNT
                            sibling_index = NodeDSAR.NOSIBLING
                            isfragment = True
                    elif ndaughters==0:
                        # This node is a leaf node. The parent node pointed to a triangle ID.
                        # So the parent_node_dict won't have our info to rope back
                        # however, we nevery will navigate here. we will traverse at the parent and put the photon in the intersection working queue
                        # however, it ought to be represented in the nodedsar tree
                        isleaf = True
                        try:
                            parent_index = parent_node_dict[node_index]['parent']
                            aunt_index   = parent_node_dict[node_index]['aunt']
                            if sibling_index not in parent_node_dict or parent_node_dict[sibling_index]['parent']!=parent_index:
                                sibling_index = NodeDSAR.NOSIBLING # actually a cousin
                        except:
                            parent_index = NodeDSAR.NOPARENT
                            aunt_index   = NodeDSAR.NOAUNT
                else:
                    parent_index = NodeDSAR.NOPARENT
                    aunt_index   = NodeDSAR.NOAUNT

                    
                # Create node data
                nodedsar = NodeDSAR( node_index, parent=parent_index, aunt=aunt_index, sibling=sibling_index, fdaughter=daughter_index, ndaughters=node['nchild'] )
                if isleaf:
                    nodedsar.isleaf()
                if isfragment:
                    nodedsar.isfragment()
                
                # store in tree    
                self.dsartree[node_index] = nodedsar

            parent_node_dict = self._make_parent_node_dict( layer_nodes, layer_start, layer_end, nodes )

            #print "Parent Dict for next layer: ",parent_node_dict
            #self.printlayer( layer_start, layer_end )

    def _make_arrays(self):
        nnodes = len(self.dsartree)
        self.node_index     = np.arange( nnodes, dtype=np.uint32 )
        self.parent         = np.zeros( nnodes, dtype=np.uint32 )
        self.first_daughter = np.zeros( nnodes, dtype=np.uint32 )
        self.ndaughters     = np.zeros( nnodes, dtype=np.uint32 )
        self.sibling        = np.zeros( nnodes, dtype=np.uint32 )
        self.aunt           = np.zeros( nnodes, dtype=np.uint32 )
        for n in xrange(0,nnodes):
            self.node_index[n] = self.dsartree[n].node_index
            self.parent[n] = self.dsartree[n].parent
            self.first_daughter[n] = self.dsartree[n].first_daughter
            self.ndaughters[n] = self.dsartree[n].ndaughters
            self.sibling[n] = self.dsartree[n].sibling
            self.aunt[n] = self.dsartree[n].aunt

    def _make_parent_node_dict(self, layer_nodes, layer_start, layer_end, nodes ):
        parent_node_dict = {}
        for inode, node in enumerate(layer_nodes):
            node_index    = np.uint32(layer_start + inode)
            aunt_index    = node_index + np.uint32(1)
            if aunt_index>=np.uint32(layer_end):
                aunt_index = NodeDSAR.NOAUNT
            nchild = node['nchild']
            if nchild>0:
                # daughter is internal node
                for daughter_index in xrange(node['child'],node['child']+node['nchild']):
                    parent_node_dict[ daughter_index ] = {'parent':node_index, 'aunt':aunt_index}
            elif nchild==0:
                pass
        return parent_node_dict

    def printlayer(self,layer_start, layer_end):
        for inode in xrange(layer_start,layer_end):
            print str(self.dsartree[inode])

