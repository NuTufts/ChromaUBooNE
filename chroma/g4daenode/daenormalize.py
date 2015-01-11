import collada

class DAENormalize(object):
    misordered = ["{%s}instance_node" % COLLADA_NS, "{%s}matrix" % COLLADA_NS ]
    materials_ptn = re.compile("__dd__Materials__(\S*)_fx_0x\S{7}$")
    def __init__(self, xmlnode ):
        nodes = xmlnode.xpath("//c:node", namespaces=NAMESPACES)
        self.normalize_nodes( nodes )
        effects = xmlnode.xpath("//c:effect", namespaces=NAMESPACES)
        self.normalize_effects( effects )

    def normalize_nodes(self, nodes):
        """
        For unknown reasons small numbers of nodes have the matrix at the end, 
        with mis-ordered layout::

               node 
                    instance_node
                    matrix  

        Corrected layout::

               node 
                    matrix  
                    instance_node

        """
        normnode = 0  
        for node in nodes:
            tags = map(lambda _:_.tag, node)
            if tags == self.misordered: 
                node[:] = list(reversed(list(node)))
                normnode += 1 
            pass
        pass 
        log.info("swapped %s / %s node misorders : node/[instance_node <--> matrix] " % (normnode,len(nodes)) )


    colormap = {
                 'MineralOil':'1.0 0.0 0.0 1.0',
                    'Acrylic':'0.5 0.5 0.5 1.0',
         'LiquidScintillator':'0.0 1.0 0.0 1.0',
                  'GdDopedLS':'0.0 0.0 1.0 1.0',
                     'Teflon':'1.0 0.5 0.5 1.0',
                      'Pyrex':'0.0 0.0 1.0 1.0',
                     'Vacuum':'1.0 0.5 0.5 1.0',
                   'Bialkali':'1.0 0.0 0.0 1.0',
               'OpaqueVacuum':'1.0 0.5 0.5 1.0',
         'UnstStainlessSteel':'1.0 0.5 0.5 1.0',
                        'PVC':'1.0 0.5 0.5 1.0',
             'StainlessSteel':'0.5 0.5 0.5 1.0',
                        'Air':'1.0 0.5 0.5 1.0',
                        'ESR':'1.0 0.0 0.0 1.0',
                      'Nylon':'1.0 0.5 0.5 1.0',
     }


    transmap = {
                 'MineralOil':'0.1',
                    'Acrylic':'0.1',
         'LiquidScintillator':'0.2',
                  'GdDopedLS':'0.3',
                     'Teflon':'0.0',
                      'Pyrex':'0.4',
                     'Vacuum':'0.0',
                   'Bialkali':'0.5',
               'OpaqueVacuum':'0.0',
         'UnstStainlessSteel':'0.2',
                        'PVC':'0.0',
             'StainlessSteel':'0.2',
                        'Air':'0.0',
                        'ESR':'0.1',
                      'Nylon':'0.0',
 

    }
 

    def normalize_effects(self, effects):
        """
        Change colors for effects which have id matching the map


        phong
              Produces a specularly shaded surface where the specular reflection 
              is shaded according the Phong BRDF approximation.
              emission/ambient/diffuse/*specular/shininess*/reflective/reflectivity/transparent/transparency/index_of_refraction

        lambert
              Produces a diffuse shaded surface that is independent of lighting.
              emission/ambient/diffuse/reflective/reflectivity/transparent/transparency/index_of_refraction


        """
        for effect in effects:
            id_ = effect.attrib['id']
            match = self.materials_ptn.match(id_)
            if match: 
                name = match.groups()[0]
                coltxt = self.colormap.get(name, None)
                if not coltxt is None:
                    log.info("effects %s %s " % (name, coltxt) ) 
                    for color in effect.findall(".//c:color", namespaces=NAMESPACES):
                        color.text = coltxt
                        #color.text = '0 0 0 1'
                    pass
                    #emission_color = effect.find(".//c:emission/c:color", namespaces=NAMESPACES)   
                    #emission_color.text = coltxt
                    pass


                transparency = effect.find(".//c:transparency/c:float", namespaces=NAMESPACES)
                if not transparency is None:
                    tratxt = self.transmap.get(name, '0.1')   # transparency of 1. is fully non-transparent, transparency 0. is fully-transparent and hence invisible
                    transparency.text = tratxt   
                pass    

                technique = effect.find(".//c:technique", namespaces=NAMESPACES )

                # "phong" and "lambert" allowed children differ, 
                # remove the phong disallowed 
                #
                phong = technique.find("./c:phong", namespaces=NAMESPACES )
                specular = phong.find("./c:specular", namespaces=NAMESPACES )
                shininess = phong.find("./c:shininess", namespaces=NAMESPACES )
                phong.remove(specular)
                phong.remove(shininess) 
                phong.tag = '{%s}lambert' % COLLADA_NS

                pass

