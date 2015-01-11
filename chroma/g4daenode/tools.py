from collada.xmlutil import etree as ET
from collada.xmlutil import writeXML, COLLADA_NS, E
NAMESPACES=dict(c=COLLADA_NS)
tag = lambda _:str(ET.QName(COLLADA_NS,_))

def remove_address( identifier ):
    ixpo = identifier.find("0x")
    if ixpo > -1:
        identifier = identifier[:ixpo]
    return identifier


def present_geometry( bg ):
    out = []
    out.append(bg)
    for bp in bg.primitives():
        out.append("nvtx:%s" % len(bp.vertex))
        out.append(bp.vertex)
    return out

def read_properties( xmlnode ):
    """ Reads in properties from XML node"""
    data = {}       
    for matrix in xmlnode.findall(tag("matrix")):
        xref = matrix.attrib['name']
        assert matrix.attrib['coldim'] == '2' 
        data[xref] = as_optical_property_vector( matrix.text )
    pass
    properties = {} 
    for property_ in xmlnode.findall(tag("property")):
        prop = property_.attrib['name']
        xref = property_.attrib['ref']
        #assert xref in data   # failing for LXe 
        if xref in data:
            properties[prop] = data[xref] 
        else:
            log.warn("xref not in data for property_ %s " % repr(property_))
        pass
    return properties
