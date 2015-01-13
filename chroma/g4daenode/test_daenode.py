from chroma.g4daenode import DAENode
#from chroma.g4daenode.collada_to_chroma import daeload
from collada_to_chroma import daeload
import logging


if __name__ == "__main__":
    #testfile = "microboone_nowires.dae" # MicroBooNE geometry with no wires
    testfile = "../../gdml/microboone_nowires_chroma_simplified.dae" # Restructured to be more Collada/Chroma friendly
    logging.basicConfig(filename='log.test_daenode',level=logging.DEBUG)
    print "Running test file: ",testfile
    DAENode.parse( testfile, sens_mats=[] )

    chroma_geom, c2c = daeload( testfile )
    print chroma_geom

    lar = c2c.materialmap['LAr']
    print chroma_geom.mesh.vertices
    print chroma_geom.mesh.triangles
    print len(chroma_geom.mesh.triangles)
    for mat in  chroma_geom.unique_materials:
        print "[",mat.name,"]"
        print mat.refractive_index
        print mat.absorption_length
        print mat.scattering_length
        print mat.reemission_prob
        print mat.reemission_cdf
        raw_input()
    print chroma_geom.unique_surfaces
    print chroma_geom.material1_index
    print len(chroma_geom.material1_index)
