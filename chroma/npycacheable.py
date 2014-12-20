#!/usr/bin/env python
"""
NPYCache
========

Introspective single file NPY cache machinery. 

"""
import logging, os
import importlib
log = logging.getLogger(__name__)
import types
import numpy as np


class NPYCacheable(object):
    """
    Object cacheing as directory hierarchy 
    of single array .npy files

    Motivation for extreme simplicity of structure 
    is to facilitate reading from C/C++

    See *env/chromacpp-* for investigation of
    reading objects cached with this into C/C++
    """
    primitives_name = "__primitives.dict"
    none_name = "__none.txt"



    @classmethod
    def compose_instance_label(cls, obj):
        if not getattr(obj,'name', None) is None:  
            identifier = getattr(obj, 'name')
        elif not getattr(obj,'id', None) is None:  
            identifier = getattr(obj, 'id')
        else:
            identifier = "0x%x" % (id(obj))
        pass
        label = "%s:%s:%s" % (obj.__class__.__module__,obj.__class__.__name__, identifier)
        return label

    @classmethod
    def decompose_instance_label(cls, label):
        elem = label.split(":")
        assert len(elem) == 3, elem
        return elem 
        
    @classmethod
    def represents_object(cls, base):
        """
        Directory with one subdir represents an object 
        """
        basepath = os.path.join(*base)
        names = os.listdir(basepath)
        return len(names) == 1 

    @classmethod
    def represents_ndarray_item(cls, base):
        """
        Directory with one .npy file represents an ndarray
        as item of a sequence 
        """
        basepath = os.path.join(*base)
        names = os.listdir(basepath)
        return len(names) == 1 and names[0].endswith(".npy")

    @classmethod
    def load_ndarray_item(cls, base):
        basepath = os.path.join(*base)
        assert cls.represents_ndarray_item(base), basepath
        names = os.listdir(basepath)
        return np.load(os.path.join(basepath,names[0]))

    @classmethod
    def represents_sequence(cls, base):
        """
        Directory containing subdirs with numerical labels
        represents a list or tuple 
        """
        basepath = os.path.join(*base)
        indices = []
        names = os.listdir(basepath)
        for name in names:
            try:
                indices.append(int(name))  
            except ValueError:
                pass
            pass    
        pass
        isseq = len(indices) == len(names)
        return isseq

    @classmethod
    def load_sequence(cls, base):
        assert cls.represents_sequence(base)
        log.debug("create_list %s " % os.path.join(*base))
        obj = list()
        for name in os.listdir(os.path.join(*base)):
            instance = cls.load_instance(base+[name])
            obj.append(instance)
        return obj

    @classmethod
    def save_sequence(cls, base, obj):
        assert type(obj) in (list,tuple), "not list or tuple" 
        if len(obj) > 999:
            log.warn("skip list/tuple thats too long %s %s " % (len(obj), os.path.join(*base)))
            return 

        log.info("save_list %s len %s " % (os.path.join(*base), len(obj)))
        for i in range(len(obj)): 
            item = obj[i]
            idxref = base + ["%0.3d" % i]
            idxdir = os.path.join(*idxref)
            if not os.path.exists(idxdir):
                os.makedirs(idxdir)
            pass
            if isinstance(item, NPYCacheable):
                item.save(idxref)
            elif type(item) == np.ndarray:
                cls.save_ndarray(idxref + ["item"], item)
            elif item is None:
                cls.save_none(idxref)
            else:
                log.warn("skipping sequence element %s %s " % (i, obj[i])) 
        pass



    @classmethod
    def save_ndarray(cls, base, obj):
        assert type(obj) == np.ndarray, "not ndarray" 
        path = os.path.join(*base) + ".npy" 
        log.debug("saving ndarray to %s shape %s  " % (path, obj.shape) )
        sdir = os.path.dirname(path)
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        pass
        np.save( path, obj ) 


    @classmethod
    def primitives_path(cls, base):
        elem = base + [cls.primitives_name]
        return os.path.join(*elem)

    @classmethod
    def save_primitives(cls, base, obj):
        assert type(obj) == dict, "not dict" 
        path = cls.primitives_path(base)
        with open(path, "w") as fp:
            fp.write(repr(obj))

    @classmethod
    def load_primitives(cls, base):
        path = cls.primitives_path(base)
        with open(path, "r") as fp:
            txt = fp.read()
        assert txt[0] == '{' and txt[-1] == '}'
        return eval(txt)




    @classmethod
    def none_path(cls, base):
        elem = base + [cls.none_name]
        return os.path.join(*elem)

    @classmethod
    def save_none(cls, base):
        path = cls.none_path(base)
        with open(path,"w") as fp:
            fp.write("none")
        
    @classmethod
    def load_none(cls, base):
        path = cls.none_path(base)
        assert os.path.exists(path), path
        return None

    @classmethod
    def represents_none(cls, base):
        path = cls.none_path(base)
        return os.path.exists(path)

 
    @classmethod
    def load_object(cls, base):
        assert cls.represents_object(base)
        log.debug("create_object %s " % os.path.join(*base))
        label = os.listdir(os.path.join(*base))[0]   # only one dir inside
        base.append(label)

        modname, klsname, identifier = cls.decompose_instance_label(label)
        mod = importlib.import_module(modname)
        kls = getattr(mod, klsname) 
        obj = kls()

        constituent = {}
        for name in os.listdir(os.path.join(*base)):
            elem = base + [name]
            path = os.path.join(*elem)
            if not os.path.isdir(path):
                if path.endswith('.npy'):
                    constituent[name[:-4]] = np.load(path)  
                else:
                    pass
            else:
                constituent[name] = cls.load_instance(elem)
            pass
        for k,v in constituent.items():
            setattr(obj, k, v )
        pass
         
        primitives = cls.load_primitives(base)
        for k,v in primitives.items():
            setattr(obj, k, v)
        pass

        return obj

    @classmethod
    def load_instance(cls, base):
        """
        #. the order of directory sniffing needs to be from most to least specific 
        """
        if type(base) == str:
            base=[base]

        log.debug("create_instance %s " % os.path.join(*base))
        
        if cls.represents_ndarray_item(base):
            instance = cls.load_ndarray_item(base) 
        elif cls.represents_none(base):
            instance = cls.load_none(base) 
        elif cls.represents_sequence(base):
            instance = cls.load_sequence(base)
        elif cls.represents_object(base):
            instance = cls.load_object(base)
        else:
            assert 0 , "failed to recognize what dir %s represents " % os.path.join(*base)
        return instance
 
    @classmethod
    def skip_attribute(cls, name):
        cache_skips = getattr(cls, 'cache_skips', [])
        return name in cache_skips

    def save(self, base):
        """
        :param base:  List to be os.path.join-ed to form directory path at which to save
        """
        if type(base) == str:
            base = [base]

        label = self.compose_instance_label(self)
        elem = base + [label]

        primitives = {}
        for attn in filter(lambda _:_[0] != "_",dir(self)):
            obj = getattr(self,attn)
            alem = elem + [attn]

            if type(obj) == types.MethodType:
                continue
            elif self.skip_attribute(attn):
                log.info("skip_attribute %s " % attn )
                continue 
            elif type(obj) == list:
                self.save_sequence(alem, obj)
            elif isinstance(obj, tuple):
                self.save_sequence(alem, obj)
            elif type(obj) == np.ndarray:
                self.save_ndarray(alem, obj)
            elif isinstance(obj, NPYCacheable):
                obj.save(alem)
            elif isinstance(obj, str):
                primitives[attn] = obj
            elif isinstance(obj, int):
                primitives[attn] = obj
            elif isinstance(obj, float):
                primitives[attn] = obj
            elif obj is None:
                primitives[attn] = obj
            elif isinstance(obj, dict):  # check contains primitives only?  
                primitives[attn] = obj
            else:
                log.info("skip %s type %s  %s " % (attn, type(obj), repr(obj))) 
            pass
        pass
        self.save_primitives(elem, primitives)



if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.DEBUG)
    import IPython

    container = "/tmp/env/chroma_geometry"
    d = NPYCacheable.load_instance(container)

    IPython.embed()

