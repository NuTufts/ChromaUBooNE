import collada

class DAESubTree(list):
    """
    Flattens all or part of a tree of nodes into this list. 

    Only requires node instances to have a children attribute 
    which lists other nodes. The list is composed of either:

    #. a string representation of the node
    #. a tuple of (node, depth, sibdex, indent) 
    """
    def __init__(self, top, maxdepth=-1, text=True, maxsibling = 5):
        """
        :param top: root node instance such as `DAENode`  
        :param maxdepth: integer maximum recursion depth, default of -1 for unlimited
        :param text: when True makes makes list-of-strings representation of tree, otherwise
                     make list of tuples incoporating the node in first slot of each tuple

        :param maxsibling: siblings are skipped when the number of children
                           of a node exceeds 2*maxsibling, 
                           the first and last `maxsiblings` are incorporated into this list
                           
        """
        list.__init__(self)
        self.maxdepth = maxdepth
        self.text = text
        self.cut = maxsibling 
        self( top )

    __str__ = lambda _:"\n".join(_)

    def __call__(self, node, depth=0, sibdex=-1, nsibling=-1 ):
        """
        :param node:  
        :param depth:
        :param sibdex: sibling index from 0:nsibling-1
        :param nsibling: 
        """
        if not hasattr(node,'children'):
            nchildren = 0  
        else:
            nchildren = len(node.children) 
        pass
        elided = type(node) == str
        indent = "   " * depth     # done here as difficult to do in a webpy template
        if self.text:
            if elided:
                obj = "..."
            else:
                nodelabel = "%-2d %-5d %-3d" % (depth, node.index, nchildren )
                obj = "[%s] %s %3d/%3d : %s " % (nodelabel, indent, sibdex, nsibling, node)
        else:
            obj = (node, depth, sibdex, indent)
        pass     
        self.append( obj )


        if nchildren == 0:# leaf
            pass
        else:
            if depth == self.maxdepth:
                pass
            else:    
                shorten = nchildren > self.cut*2    
                for sibdex, child in enumerate(node.children):
                    if shorten:
                        if sibdex < self.cut or sibdex > nchildren - self.cut:
                            pass
                        elif sibdex == self.cut:    
                            child = "..."
                        else:
                            continue
                    pass         
                    self(child, depth + 1, sibdex, nchildren )

