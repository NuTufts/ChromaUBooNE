import traceback
import numpy as np
import chroma.api as api
if api.is_gpu_api_cuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray as ga
    from pycuda import characterize
    import chroma.gpu.cutools as cutools
elif api.is_gpu_api_opencl():
    import pyopencl as cl
    import pyopencl.array as ga
    import chroma.gpu.cltools as cltools
else:
    raise RuntimeError('API neither CUDA or OpenCL')

from collections import OrderedDict

from chroma.geometry import standard_wavelengths
from chroma.gpu.tools import chunk_iterator, format_array, format_size, to_uint3, to_float3, \
    make_gpu_struct, mapped_empty, Mapped, copy_to_float3, copy_to_uint3
from chroma.gpu.gpufuncs import GPUFuncs

#from chroma.log import logger
import logging
log = logging.getLogger(__name__)

# This class is responsible for packing the geometry description found 
#    in the chroma.geometry.Geometry class onto the GPU.
# Unfortunatey, how this is done will be very different for OpenCL and CUDA
# Had to break the constructor up into pieces

class Metadata(OrderedDict):
    def __init__(self):
        OrderedDict.__init__(self)

    def array(self, prefix, arr):
        self['%s_nbytes' % prefix] = arr.nbytes
        try:
            self['%s_itemsize' % prefix] = arr.itemsize
        except:
            pass
        self['%s_count' % prefix] = len(arr)

    def __call__(self, tag, description):
        if api.is_gpu_api_cuda():
            gpu_free, gpu_total = cuda.mem_get_info() 
        elif api.is_gpu_api_opencl():
            ctx = cltools.get_last_context()
            device = ctx.get_info( cl.context_info.DEVICES )[0]
            gpu_total = device.get_info( cl.device_info.GLOBAL_MEM_SIZE )
            gpu_free = gpu_total # free memory info not availabe to opencl...
        if tag is None:
            self['gpu_total'] = gpu_total 
        else:
            self['%s' % tag ] = description
            self['%s_gpu_used' % tag] = gpu_total - gpu_free 
        pass


class GPUGeometry(object):
    instance_count = 0
    def __init__(self, geometry, wavelengths=None, print_usage=False, min_free_gpu_mem=300e6, cl_context=None, cl_queue=None):
        log.info("GPUGeometry.__init__ min_free_gpu_mem %s ", min_free_gpu_mem)

        self.instance_count += 1
        assert self.instance_count == 1,  traceback.print_stack()

        self.metadata = Metadata()
        self.metadata(None,'preinfo')
        self.metadata('a',"start")
        self.metadata['a_min_free_gpu_mem'] = min_free_gpu_mem 

        if wavelengths is None:
            self.wavelengths = standard_wavelengths
        else:
            self.wavelengths = wavelengths

        try:
            self.wavelength_step = np.unique(np.diff(self.wavelengths)).item()
        except ValueError:
            raise ValueError('wavelengths must be equally spaced apart.')

        # this is where things get difficult.
        # pycuda and pyopencl gives us very different methods for working with structs
        #geometry_struct_size = characterize.sizeof('Geometry', geometry_source)

        # Note, that unfortunately the data types returned are very different as the 
        if api.is_gpu_api_cuda():
            self.material_data, self.material_ptrs, self.material_pointer_array = self._package_material_data_cuda( geometry, self.wavelengths, self.wavelength_step )
            self.surface_data, self.surface_ptrs, self.surface_pointer_array    = self._package_surface_data_cuda( geometry, self.wavelengths, self.wavelength_step )
        elif api.is_gpu_api_opencl():
            self.material_data, materials_bytes_cl = self._package_material_data_cl( cl_context, cl_queue, geometry, self.wavelengths, self.wavelength_step )
            self.surface_data, surfaces_bytes_cl   = self._package_surface_data_cl( cl_context, cl_queue, geometry, self.wavelengths, self.wavelength_step )

        self.metadata('b', "after materials,surfaces") 
        if api.is_gpu_api_opencl():
            self.metadata['b_gpu_used'] = materials_bytes_cl+surfaces_bytes_cl # opencl, we have to track this ourselves

        # Load Vertices and Triangles
        if api.is_gpu_api_cuda():
            self.vertices   = mapped_empty(shape=len(geometry.mesh.vertices),
                                         dtype=ga.vec.float3,
                                         write_combined=True)
            self.vertices4  = np.zeros( shape=( len(self.vertices), 4 ), dtype=np.float32 )
            self.triangles  = mapped_empty(shape=len(geometry.mesh.triangles),
                                          dtype=ga.vec.uint3,
                                          write_combined=True)
            self.triangles4 = np.zeros( shape=( len(self.triangles), 4 ), dtype=np.uint32 )
            self.vertices[:]       = to_float3(geometry.mesh.vertices)
            self.vertices4[:,:-1]  = self.vertices.ravel().view( np.float32 ).reshape( len(self.vertices), 3 )  # for textures
            self.triangles[:]      = to_uint3(geometry.mesh.triangles)
            self.triangles4[:,:-1] = self.triangles.ravel().view( np.uint32 ).reshape( len(self.triangles), 3 ) # for textures
        elif api.is_gpu_api_opencl():
            self.vertices = ga.empty( cl_queue, len(geometry.mesh.vertices), dtype=ga.vec.float3 )
            self.triangles = ga.empty( cl_queue, len(geometry.mesh.triangles), dtype=ga.vec.uint3 )
            self.vertices[:] = to_float3( geometry.mesh.vertices )
            self.triangles[:] = to_uint3( geometry.mesh.triangles )
        
        if api.is_gpu_api_cuda():
            self.world_origin = ga.vec.make_float3(*geometry.bvh.world_coords.world_origin)
        elif api.is_gpu_api_opencl():
            self.world_origin = ga.vec.make_float3(*geometry.bvh.world_coords.world_origin)
            #self.world_origin = geometry.bvh.world_coords.world_origin
            self.world_origin = ga.to_device( cl_queue, self.world_origin )
            print type(self.world_origin),self.world_origin
        self.world_scale = np.float32(geometry.bvh.world_coords.world_scale)

        # Load material and surface indices into 8-bit codes
        # check if we've reached a complexity threshold
        if len(geometry.unique_materials)>=int(0xff):
            raise ValueError('Number of materials to index has hit maximum of %d'%(int(0xff)))
        if len(geometry.unique_surfaces)>=int(0xff):
            raise ValueError('Number of surfaces to index has hit maximum of %d'%(int(0xff)))
        # make bit code
        material_codes = (((geometry.material1_index & 0xff) << 24) |
                          ((geometry.material2_index & 0xff) << 16) |
                          ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
        if api.is_gpu_api_cuda():
            self.material_codes = ga.to_gpu(material_codes)
        elif api.is_gpu_api_opencl():
            self.material_codes = ga.to_device(cl_queue,material_codes)

        # assign color codes
        colors = geometry.colors.astype(np.uint32)
        if api.is_gpu_api_cuda():
            self.colors = ga.to_gpu(colors)
            self.solid_id_map = ga.to_gpu(geometry.solid_id.astype(np.uint32))
        elif api.is_gpu_api_opencl():
             self.colors = ga.to_device(cl_queue,colors)
             self.solid_id_map = ga.to_device(cl_queue,geometry.solid_id.astype(np.uint32))


        # Limit memory usage by splitting BVH into on and off-GPU parts
        self.metadata('c', "after colors, idmap") 
        if api.is_gpu_api_cuda():
            gpu_free, gpu_total = cuda.mem_get_info()
        elif api.is_gpu_api_opencl():
            gpu_total = self.metadata['gpu_total']
            meshdef_nbytes_cl = self.vertices.nbytes+self.triangles.nbytes+self.world_origin.nbytes+self.world_scale.nbytes+self.material_codes.nbytes+self.colors.nbytes+self.solid_id_map.nbytes
            self.metadata['c_gpu_used'] = materials_bytes_cl+surfaces_bytes_cl+meshdef_nbytes_cl
            gpu_free = gpu_total - (materials_bytes_cl+surfaces_bytes_cl+meshdef_nbytes_cl)


        # Figure out how many elements we can fit on the GPU,
        # but no fewer than 100 elements, and no more than the number of actual nodes
        n_nodes = len(geometry.bvh.nodes)
        split_index = min(
            max(int((gpu_free - min_free_gpu_mem) / geometry.bvh.nodes.itemsize),100),
            n_nodes
            )
        print "split index=",split_index," vs. total nodes=",n_nodes
 
        # push nodes to GPU
        if api.is_gpu_api_cuda():
            self.nodes = ga.to_gpu(geometry.bvh.nodes[:split_index])
        elif api.is_gpu_api_opencl():
            self.nodes = ga.to_device(cl_queue,geometry.bvh.nodes[:split_index])
        n_extra = max(1, (n_nodes - split_index)) # forbid zero size

        # left over nodes
        if api.is_gpu_api_cuda():
            self.extra_nodes = mapped_empty(shape=n_extra,
                                            dtype=geometry.bvh.nodes.dtype,
                                            write_combined=True)
        elif api.is_gpu_api_opencl():
            self.extra_nodes = ga.empty( cl_queue, shape=n_extra, dtype=geometry.bvh.nodes.dtype )

        if split_index < n_nodes:
            log.info('Splitting BVH between GPU and CPU memory at node %d' % split_index)
            self.extra_nodes[:] = geometry.bvh.nodes[split_index:]
            splitting = 1
        else:
            splitting = 0

        self.metadata('d',"after nodes")
        if api.is_gpu_api_opencl():
            nodes_nbytes_cl = self.nodes.nbytes
            self.metadata['d_gpu_used'] = materials_bytes_cl+surfaces_bytes_cl+meshdef_nbytes_cl+nodes_nbytes_cl
        self.metadata.array("d_nodes", geometry.bvh.nodes )
        self.metadata['d_split_index'] = split_index
        self.metadata['d_extra_nodes_count'] = n_extra
        self.metadata['d_splitting'] = splitting
        self.print_device_usage(cl_context=cl_context)

        # CUDA See if there is enough memory to put the vertices and/or triangles back on the GPU
        if api.is_gpu_api_cuda():
            gpu_free, gpu_total = cuda.mem_get_info()
        elif api.is_gpu_api_opencl():
            gpu_total = self.metadata['gpu_total']
            gpu_free = gpu_total-self.metadata['d_gpu_used']
        self.metadata.array('e_triangles', self.triangles)
        if api.is_gpu_api_cuda():
            if self.triangles.nbytes < (gpu_free - min_free_gpu_mem):
                self.triangles = ga.to_gpu(self.triangles)
                log.info('Optimization: Sufficient memory to move triangles onto GPU')
                ftriangles_gpu = 1
            else:
                log.warn('using host mapped memory triangles')
                ftriangles_gpu = 0
        elif api.is_gpu_api_opencl():
            if self.triangles.nbytes < (gpu_free - min_free_gpu_mem):
                #self.triangles = ga.to_device(cl_queue,self.triangles)
                log.info('Optimization: Sufficient memory to move triangles onto GPU')
                ftriangles_gpu = 1
            else:
                log.warn('using host mapped memory triangles')
                ftriangles_gpu = 0

        self.metadata('e',"after triangles")
        self.metadata['e_triangles_gpu'] = ftriangles_gpu

        if api.is_gpu_api_cuda():
            gpu_free, gpu_total = cuda.mem_get_info()
        elif api.is_gpu_api_opencl():
            gpu_total = self.metadata['gpu_total']
            gpu_free = gpu_total-self.metadata['d_gpu_used']

        self.metadata.array('f_vertices', self.vertices )

        if api.is_gpu_api_cuda():
            if self.vertices.nbytes < (gpu_free - min_free_gpu_mem):
                self.vertices = ga.to_gpu(self.vertices)
                log.info('Optimization: Sufficient memory to move vertices onto GPU')
                vertices_gpu = 1
            else:
                log.warn('using host mapped memory vertices')
                vertices_gpu = 0
        elif api.is_gpu_api_opencl():
            if self.vertices.nbytes < (gpu_free - min_free_gpu_mem):
                #self.vertices = ga.to_gpu(self.vertices)
                log.info('Optimization: Sufficient memory to move vertices onto GPU')
                vertices_gpu = 1
            else:
                log.warn('using host mapped memory vertices')
                vertices_gpu = 0

        self.metadata('f',"after vertices")
        self.metadata['f_vertices_gpu'] = vertices_gpu
        
        if api.is_gpu_api_cuda():
            geometry_source = cutools.get_cu_source('geometry_types.h')
            geometry_struct_size = characterize.sizeof('Geometry', geometry_source)
            self.gpudata = make_gpu_struct(geometry_struct_size,
                                           [Mapped(self.vertices), 
                                            Mapped(self.triangles),
                                            self.material_codes,
                                            self.colors, self.nodes,
                                            Mapped(self.extra_nodes),
                                            self.material_pointer_array,
                                            self.surface_pointer_array,
                                            self.world_origin,
                                            self.world_scale,
                                            np.int32(len(self.nodes))])
            self.geometry = geometry
        elif api.is_gpu_api_opencl():
            # No relevant way to pass struct into OpenCL kernel. We have to pass everything by arrays
            # We then build a geometry struct later in the kernel
            # provided below is example/test of passing the data
            #if True: # for debuggin
            if False: #
                print "loading geometry_structs.cl"
                geostructsmod = cltools.get_cl_module( "geometry_structs.cl", cl_context, options=cltools.cl_options, include_source_directory=True )
                geostructsfunc = GPUFuncs( geostructsmod )
                geostructsfunc.make_geostruct( cl_queue, (3,), None,
                                               self.vertices.data, self.triangles.data,
                                               self.material_codes.data, self.colors.data,
                                               self.nodes.data, self.extra_nodes.data,
                                               np.int32(len(geometry.unique_materials)),
                                               self.material_data['refractive_index'].data, self.material_data['absorption_length'].data,
                                               self.material_data['scattering_length'].data, self.material_data['reemission_prob'].data,
                                               self.material_data['reemission_cdf'].data,
                                               np.int32(len(geometry.unique_surfaces)),
                                               self.surface_data['detect'].data, self.surface_data['absorb'].data, self.surface_data['reemit'].data,
                                               self.surface_data['reflect_diffuse'].data, self.surface_data['reflect_specular'].data,
                                               self.surface_data['eta'].data, self.surface_data['k'].data, self.surface_data['reemission_cdf'].data,
                                               self.surface_data['model'].data, self.surface_data['transmissive'].data, self.surface_data['thickness'].data,
                                               self.world_origin, self.world_scale, np.int32( len(self.nodes) ),
                                               self.material_data['n'], self.material_data['step'], self.material_data["wavelength0"] )
                cl_queue.finish()
                self.material_codes.get()
                raise RuntimeError('bail')
        if print_usage:
            self.print_device_usage(cl_context=cl_context)
        log.info(self.device_usage_str(cl_context=cl_context))
        self.metadata('g',"after geometry struct")


    def _interp_material_property( self, wavelengths, property ):
        # note that it is essential that the material properties be
        # interpolated linearly. this fact is used in the propagation
        # code to guarantee that probabilities still sum to one.
        return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)

    def _package_material_data_cuda(self, geometry, wavelengths, wavelength_step ):
        material_data = []
        material_ptrs = []
        geometry_source = cutools.get_cu_source('geometry_types.h')
        material_struct_size = characterize.sizeof('Material', geometry_source)
        
        for i in range(len(geometry.unique_materials)):
            material = geometry.unique_materials[i]

            if material is None:
                raise Exception('one or more triangles is missing a material.')

            refractive_index      = self._interp_material_property(wavelengths, material.refractive_index)
            refractive_index_gpu  = ga.to_gpu(refractive_index)
            absorption_length     = self._interp_material_property(wavelengths, material.absorption_length)
            absorption_length_gpu = ga.to_gpu(absorption_length)
            scattering_length     = self._interp_material_property(wavelengths, material.scattering_length)
            scattering_length_gpu = ga.to_gpu(scattering_length)
            reemission_prob       = self._interp_material_property(wavelengths, material.reemission_prob)
            reemission_prob_gpu   = ga.to_gpu(reemission_prob)
            reemission_cdf        = self._interp_material_property(wavelengths, material.reemission_cdf)
            reemission_cdf_gpu = ga.to_gpu(reemission_cdf)

            material_data.append(refractive_index_gpu)
            material_data.append(absorption_length_gpu)
            material_data.append(scattering_length_gpu)
            material_data.append(reemission_prob_gpu)
            material_data.append(reemission_cdf_gpu)

            material_gpu = \
                make_gpu_struct(material_struct_size,
                                [refractive_index_gpu, absorption_length_gpu,
                                 scattering_length_gpu,
                                 reemission_prob_gpu,
                                 reemission_cdf_gpu,
                                 np.uint32(len(wavelengths)),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0])])

            material_ptrs.append(material_gpu)

        material_pointer_array = make_gpu_struct(8*len(material_ptrs), material_ptrs)
        return material_data, material_ptrs, material_pointer_array

    def _package_material_data_cl(self, context, queue, geometry, wavelengths, wavelength_step ):
        material_data = {}
        material_ptrs = None
        device = context.devices[0]

        # First thing is to define the geometry struct
        # would be great if something like this worked. But its not going to for now.
        # we have to settle for keeping track of simple arrays and passing them in as arguments to cl kernel functions...
        #material_struct = np.dtype( [('refractive_index',  np.float32,len(wavelengths)),
        #                             ('absorption_length', np.float32,len(wavelengths)),
        #                             ('scattering_length', np.float32,len(wavelengths)),
        #                             ('reemission_prob',   np.float32,len(wavelengths)),
        #                             ('reemission_cdf',    np.float32,len(wavelengths)),
        #                             ('n',                 np.uint32),
        #                             ('step',              np.float32),
        #                             ('wavelength0',       np.float32)] )
        #material_struct, material_c_decl = cl.tools.match_dtype_to_c_struct( device, "Material", material_struct )
        #print "defined Material Struct"
        #print material_c_decl
        #material_struct = cl.tools.get_or_register_dtype( "Material", material_struct )
        #print "registered with pyopencl for context=",context," device=",device
        nmaterials = len(geometry.unique_materials)
        nwavelengths = len(wavelengths)
        materials_refractive_index  = np.empty( (nmaterials, nwavelengths), dtype=np.float32 )
        materials_absorption_length = np.empty( (nmaterials, nwavelengths), dtype=np.float32 )
        materials_scattering_length = np.empty( (nmaterials, nwavelengths), dtype=np.float32 )
        materials_reemission_prob   = np.empty( (nmaterials, nwavelengths), dtype=np.float32 )
        materials_reemission_cdf    = np.empty( (nmaterials, nwavelengths), dtype=np.float32 )
        
        for i in range(len(geometry.unique_materials)):
            material = geometry.unique_materials[i]

            if material is None:
                raise Exception('one or more triangles is missing a material.')

            materials_refractive_index[i]  = self._interp_material_property(wavelengths, material.refractive_index)
            materials_absorption_length[i] = self._interp_material_property(wavelengths, material.absorption_length)
            materials_scattering_length[i] = self._interp_material_property(wavelengths, material.scattering_length)
            materials_reemission_prob[i]   = self._interp_material_property(wavelengths, material.reemission_prob)
            materials_reemission_cdf[i]    = self._interp_material_property(wavelengths, material.reemission_cdf)

        material_data["refractive_index"]  = ga.to_device(queue, materials_refractive_index.ravel() )
        material_data["absorption_length"] = ga.to_device(queue, materials_absorption_length.ravel() )
        material_data["scattering_length"] = ga.to_device(queue, materials_scattering_length.ravel() )
        material_data["reemission_prob"]   = ga.to_device(queue, materials_reemission_prob.ravel() )
        material_data["reemission_cdf"]    = ga.to_device(queue, materials_reemission_cdf.ravel() )
        material_data["n"]                 = np.uint32(nwavelengths)
        material_data["step"]              = np.float32(wavelength_step)
        material_data["wavelength0"]       = np.float32(wavelengths[0])
        material_data["nmaterials"]        = np.uint32( len(geometry.unique_materials) )

        nbytes = 0
        for data in material_data:
            nbytes += material_data[data].nbytes

        return material_data, nbytes

    def _package_surface_data_cuda( self, geometry, wavelengths, wavelength_step ):
        surface_data = []
        surface_ptrs = []
        geometry_source = cutools.get_cu_source('geometry_types.h')
        surface_struct_size = characterize.sizeof('Surface', geometry_source)

        for i in range(len(geometry.unique_surfaces)):
            surface = geometry.unique_surfaces[i]

            if surface is None:
                # need something to copy to the surface array struct
                # that is the same size as a 64-bit pointer.
                # this pointer will never be used by the simulation.
                surface_ptrs.append(np.uint64(0))
                continue

            detect = self._interp_material_property(wavelengths, surface.detect)
            detect_gpu = ga.to_gpu(detect)
            absorb = self._interp_material_property(wavelengths, surface.absorb)
            absorb_gpu = ga.to_gpu(absorb)
            reemit = self._interp_material_property(wavelengths, surface.reemit)
            reemit_gpu = ga.to_gpu(reemit)
            reflect_diffuse = self._interp_material_property(wavelengths, surface.reflect_diffuse)
            reflect_diffuse_gpu = ga.to_gpu(reflect_diffuse)
            reflect_specular = self._interp_material_property(wavelengths, surface.reflect_specular)
            reflect_specular_gpu = ga.to_gpu(reflect_specular)
            eta = self._interp_material_property(wavelengths, surface.eta)
            eta_gpu = ga.to_gpu(eta)
            k = self._interp_material_property(wavelengths, surface.k)
            k_gpu = ga.to_gpu(k)
            reemission_cdf = self._interp_material_property(wavelengths, surface.reemission_cdf)
            reemission_cdf_gpu = ga.to_gpu(reemission_cdf)
            nplanes_np  = np.array( surface.nplanes, dtype=np.float32 )
            nplanes_gpu = ga.to_gpu( nplanes_np )
            wire_pitch_np = np.array( surface.wire_pitch, dtype=np.float32 )
            wire_pitch_gpu = ga.to_gpu( wire_pitch_np )
            wire_diameter_np = np.array( surface.wire_diameter, dtype=np.float32 )
            wire_diameter_gpu = ga.to_gpu( wire_diameter_np )

            surface_data.append(detect_gpu)
            surface_data.append(absorb_gpu)
            surface_data.append(reemit_gpu)
            surface_data.append(reflect_diffuse_gpu)
            surface_data.append(reflect_specular_gpu)
            surface_data.append(eta_gpu)
            surface_data.append(k_gpu)
            surface_data.append(reemission_cdf_gpu)
            surface_data.append( nplanes_gpu )
            surface_data.append( wire_pitch_gpu )
            surface_data.append( wire_diameter_gpu )

            surface_gpu = \
                make_gpu_struct(surface_struct_size,
                                [detect_gpu, absorb_gpu, reemit_gpu,
                                 reflect_diffuse_gpu,reflect_specular_gpu,
                                 eta_gpu, k_gpu, reemission_cdf_gpu,
                                 np.uint32(surface.model),
                                 np.uint32(len(wavelengths)),
                                 np.uint32(surface.transmissive),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0]),
                                 np.float32(surface.thickness)])

            surface_ptrs.append(surface_gpu)

        surface_pointer_array = make_gpu_struct(8*len(surface_ptrs), surface_ptrs)
        return surface_data, surface_ptrs, surface_pointer_array
            
    def _package_surface_data_cl( self, context, queue, geometry, wavelengths, wavelength_step ):
        surface_data = {}
        device = context.devices[0]
        nsurfaces = len(geometry.unique_surfaces)
        nwavelengths = len(wavelengths)

        detect           = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        absorb           = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        reemit           = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        reflect_diffuse  = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        reflect_specular = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        eta              = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        k                = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        reemission_cdf   = np.zeros( (nsurfaces, nwavelengths), dtype=np.float32 )
        model            = np.zeros( (nsurfaces,), dtype=np.uint32 )
        transmissive     = np.zeros( (nsurfaces,), dtype=np.uint32 )
        thickness        = np.zeros( (nsurfaces,), dtype=np.float32 )

        for i in range(len(geometry.unique_surfaces)):
            surface = geometry.unique_surfaces[i]
            if surface is None:
                continue

            detect[i] = self._interp_material_property(wavelengths, surface.detect)
            absorb[i] = self._interp_material_property(wavelengths, surface.absorb)
            reemit[i] = self._interp_material_property(wavelengths, surface.reemit)
            reflect_diffuse[i] = self._interp_material_property(wavelengths, surface.reflect_diffuse)
            reflect_specular[i] = self._interp_material_property(wavelengths, surface.reflect_specular)
            eta[i] = self._interp_material_property(wavelengths, surface.eta)
            k[i] = self._interp_material_property(wavelengths, surface.k)
            reemission_cdf[i] = self._interp_material_property(wavelengths, surface.reemission_cdf)
            model[i] = np.uint32(surface.model)
            transmissive[i] = np.uint32(surface.transmissive)
            thickness[i] = np.float32(surface.thickness)
            
        surface_data["detect"]           = ga.to_device( queue, detect.ravel() )
        surface_data["absorb"]           = ga.to_device( queue, absorb.ravel() )
        surface_data["reemit"]           = ga.to_device( queue, reemit.ravel() )
        surface_data["reflect_diffuse"]  = ga.to_device( queue, reflect_diffuse.ravel() )
        surface_data["reflect_specular"] = ga.to_device( queue, reflect_specular.ravel() )
        surface_data["eta"]              = ga.to_device( queue, eta.ravel() )
        surface_data["k"]                = ga.to_device( queue, k.ravel() )
        surface_data["reemission_cdf"]   = ga.to_device( queue, reemission_cdf.ravel() )
        surface_data["model"]            = ga.to_device( queue, model.ravel() )
        surface_data["transmissive"]     = ga.to_device( queue, transmissive.ravel() )
        surface_data["thickness"]        = ga.to_device( queue, thickness.ravel() )
        surface_data["n"]                = np.uint32(nwavelengths)
        surface_data["step"]             = np.float32(wavelength_step)
        surface_data["wavelength0"]      = np.float32(wavelengths[0])
        surface_data["nsurfaces"]        = np.uint32( len(geometry.unique_surfaces) )
        nbytes = 0
        for data in surface_data:
            nbytes += surface_data[data].nbytes

        return surface_data, nbytes
            
 
    def device_usage_str(self, cl_context=None):
        '''Returns a formatted string displaying the memory usage.'''
        s = 'device usage:\n'
        s += '-'*10 + '\n'
        #s += format_array('vertices', self.vertices) + '\n'
        #s += format_array('triangles', self.triangles) + '\n'
        s += format_array('nodes', self.nodes) + '\n'
        s += '%-15s %6s %6s' % ('total', '', format_size(self.nodes.nbytes)) + '\n'
        s += '-'*10 + '\n'
        if api.is_gpu_api_cuda():
            free, total = cuda.mem_get_info()
        elif api.is_gpu_api_opencl:
            total = cl_context.get_info( cl.context_info.DEVICES )[0].get_info( cl.device_info.GLOBAL_MEM_SIZE )
            free = total-self.metadata['d_gpu_used']
        s += '%-15s %6s %6s' % ('device total', '', format_size(total)) + '\n'
        s += '%-15s %6s %6s' % ('device used', '', format_size(total-free)) + '\n'
        s += '%-15s %6s %6s' % ('device free', '', format_size(free)) + '\n'
        return s

    def print_device_usage(self, cl_context=None):
        print self.device_usage_str( cl_context=cl_context )
        print 

    def reset_colors(self):
        self.colors.set_async(self.geometry.colors.astype(np.uint32))

    def color_solids(self, solid_hit, colors, nblocks_per_thread=64,
                     max_blocks=1024):
        solid_hit_gpu = ga.to_gpu(np.array(solid_hit, dtype=np.bool))
        solid_colors_gpu = ga.to_gpu(np.array(colors, dtype=np.uint32))

        module = get_cu_module('mesh.h', options=cuda_options)
        color_solids = module.get_function('color_solids')

        for first_triangle, triangles_this_round, blocks in \
                chunk_iterator(self.triangles.size, nblocks_per_thread,
                               max_blocks):
            color_solids(np.int32(first_triangle),
                         np.int32(triangles_this_round), self.solid_id_map,
                         solid_hit_gpu, solid_colors_gpu, self.gpudata,
                         block=(nblocks_per_thread,1,1), 
                         grid=(blocks,1))

