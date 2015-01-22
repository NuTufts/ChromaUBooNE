

# This class stores Node DAR information:
# DSAR = first Daughter, Sibling, Aunt Ropes
# Goal is to use this information to allow for stackless navigation


class NodeDSAR:
    NOPARENT = -99
    NOAUNT = -99
    NOSIBLING = -99
    NODAUGHTER = -99
    def __init__(self, node_index, parent=NOPARENT, fdaughter=NODAUGHTER, sibling=NOSIBLING, aunt=NOAUNT ):
        self.node_index     = node_index
        self.parent         = parent
        self.first_daughter = fdaughter
        self.sibling        = sibling
        self.aunt           = aunt

class NodeDSARtree:
    def __init__( self, bvh ):
        layer_bounds = bvh.layer_bounds
        nodes = bvh.nodes

        # We go layer by layer
        # (1) Get daughter nodes
        # (2) Make sure they are sorted by morton number (maybe not critical)
        # (3) assign them their sibling, first daughter and aunt number
        # (4) Go to next layer

        for ilayer in range(0,len(layer_bounds)):
            layer_start = layer_bounds[ilayer]
            if ilayer+1<len(layer_bounds):
                layer_end   = layer_bounds[ilayer+1]
            else:
                layer_end = len(nodes)
            layer_nodes = nodes[layer_start:layer_end]
            print layer_nodes
    
