# importgeo Module

This module contains tools to make preparing a geometry 
for import into Chroma as straight-forward and easy as possible.

The tool assumes that users will come at Chroma with Geant4 
geometries that have varying degrees of "Chroma-readiness". 
Therefore, this tool is here to inspect the geometry description,
verify that materials and surfaces are correctly specified.
And if there is a problem, the tool will need to provide a way
to change the geometry as needed.  Python is flexible enough 
to do this. We can also write out another "fixed" collada file.

## Common Pit-falls Discovered So Far:

* surface and materials not defined in GDML or in way 
  that G4DAE discovers
* geometry has too many sibling nodes with overlapping 
  triangles. Chroma may have to have a way to ignore 
  these and not get stuck.
* wire plane (or customized surface behavior) needs to be assigned

It might also be a fact that the user must always do some work 
to reorder the heirarchy of the geometry to be more nested 
to work with G4DAE/Collada.


