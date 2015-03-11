from chroma.geometry import Surface
import numpy as np

lar1nd_wireplane = Surface( 'lar1nd_wireplane' )
lar1nd_wireplane.nplanes = 3.0
lar1nd_wireplane.wire_pitch = 0.3
lar1nd_wireplane.wire_diameter = 0.015
lar1nd_wireplane.transmissive = 1
lar1nd_wireplane.model = Surface.SURFACE_WIREPLANE

def add_wireplane_surface( solid ):
    # function detector class will use to add a wireplane surface to the geometry
    # LAr1ND has two drift regions, so we need two planes
    # set surface for triangles on x=-2023.25 and x=2023.25 planes
    
    for n,triangle in enumerate(solid.mesh.triangles):
        #print [ solid.mesh.vertices[x] for x in triangle ] # for debug
        nxplane = 0
        for ivert in triangle:
            if solid.mesh.vertices[ivert,0]==-2023.25 or solid.mesh.vertices[ivert,0]==2023.25:
                nxplane += 1
        # if the numbr of vertices have the correct x value, we say we have the right traingle
        if nxplane==3: 
            print [ solid.mesh.vertices[x] for x in triangle ]
            solid.surface[ n ] = lar1nd_wireplane
            solid.unique_surfaces = np.unique( solid.surface )
