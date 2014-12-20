import g4gen
import multiprocessing
import numpy as np
import threading
import zmq
import uuid

class G4GeneratorProcess(multiprocessing.Process):
    def __init__(self, idnum, material, vertex_socket_address, photon_socket_address, seed=None):
        multiprocessing.Process.__init__(self)

        self.idnum = idnum
        self.material = material
        self.vertex_socket_address = vertex_socket_address
        self.photon_socket_address = photon_socket_address
        self.seed = seed
        self.daemon = True

    def run(self):
        gen = g4gen.G4Generator(self.material, seed=self.seed)
        context = zmq.Context()
        vertex_socket = context.socket(zmq.PULL)
        vertex_socket.connect(self.vertex_socket_address)
        photon_socket = context.socket(zmq.PUSH)
        photon_socket.connect(self.photon_socket_address)

        # Signal with the photon socket that we are online
        # and ready for messages.
        photon_socket.send('READY')

        while True:
            ev = vertex_socket.recv_pyobj()
            ev.photons_beg = gen.generate_photons(ev.vertices)
            #print 'type(ev.photons_beg) is %s' % type(ev.photons_beg)
            photon_socket.send_pyobj(ev)

def partition(num, partitions):
    """Generator that returns num//partitions, with the last item including
    the remainder.

    Useful for partitioning a number into mostly equal parts while preserving
    the sum.

    Examples:
        >>> list(partition(800, 3))
        [266, 266, 268]
        >>> sum(list(partition(800, 3)))
        800
    """
    step = num // partitions
    for i in xrange(partitions):
        if i < partitions - 1:
            yield step
        else:
            yield step + (num % partitions)

def vertex_sender(vertex_iterator, vertex_socket):
    for vertex in vertex_iterator:
        vertex_socket.send_pyobj(vertex)

def socket_iterator(nelements, socket):
    for i in xrange(nelements):
        yield socket.recv_pyobj()

class G4ParallelGenerator(object):
    def __init__(self, nprocesses, material, base_seed=None):
        self.material = material
        if base_seed is None:
            base_seed = np.random.randint(100000000)
        base_address = 'ipc:///tmp/chroma_'+str(uuid.uuid4())
        self.vertex_address = base_address + '.vertex'
        self.photon_address = base_address + '.photon'
        self.processes = [ G4GeneratorProcess(i, material, self.vertex_address, self.photon_address, seed=base_seed + i) for i in xrange(nprocesses) ]

        for p in self.processes:
            p.start()

        self.zmq_context = zmq.Context()
        self.vertex_socket = self.zmq_context.socket(zmq.PUSH)
        self.vertex_socket.bind(self.vertex_address)
        self.photon_socket = self.zmq_context.socket(zmq.PULL)
        self.photon_socket.bind(self.photon_address)

        self.processes_initialized = False
        
    def generate_events(self, vertex_iterator):
        if not self.processes_initialized:
            # Verify everyone is running and connected to avoid
            # sending all the events to one client.
            for i in xrange(len(self.processes)):
                msg = self.photon_socket.recv()
                assert msg == 'READY'
            self.processes_initialized = True

        # Doing this to avoid a deadlock caused by putting to one queue
        # while getting from another.
        vertex_list = list(vertex_iterator)
        sender_thread = threading.Thread(target=vertex_sender, args=(vertex_list, self.vertex_socket))
        sender_thread.start()
        return socket_iterator(len(vertex_list), self.photon_socket)
