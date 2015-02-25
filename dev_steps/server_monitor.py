import os,sys,time
import threading
import zmq
import numpy as np
from display_test import sim_event, start_app

max_events = 20
event_stack = []

context = zmq.Context()
port = "5556"
sub_socket = context.socket(zmq.SUB)
sub_socket.connect( "tcp://198.125.161.122:%s" % (port) ) # nudot
sub_socket.setsockopt(zmq.SUBSCRIBE, "EVENT") 

def fill_event_stack():
    while True:
        print "Listening for broadcast."
        frames = sub_socket.recv_multipart()
        if len(event_stack)<max_events:
            event_stack.append( frames[1] )
        else:
            event_stack.pop()
            event_stack.append( frames[1] )
        print "Received event. In event stack",len(event_stack)

last_update = time.time()
def update_event():
    print "Update daemon starting"
    global last_update
    while True:
        #if len(event_stack)>0 and time.time()-last_update>10.0:
        if len(event_stack)>0:
            print "updating event"
            tup = time.time()
            array_buffer = event_stack.pop()
            array_file = open('temp.npy','r+')
            array_file.write(array_buffer)
            array_file.seek(0)
            #a = np.load( array_file )
            sim_event( array_file )
            array_file.close()
            print "Time to update: ",time.time()-tup," secs"
            last_update = time.time()
            

        time.sleep(1)
    
tlarsoft = threading.Thread( name='larsoft_server', target=fill_event_stack )
tlarsoft.daemon=True
tupdate = threading.Thread( name='update', target=update_event )
tupdate.daemon=True

try:
    tlarsoft.start()
    tupdate.start()
    start_app()
    while tlarsoft.isAlive():
        tlarsoft.join(1)  # not sure if there is an appreciable cost to this.                                                                                                                  
except (KeyboardInterrupt, SystemExit):
    print '\n! Received keyboard interrupt, quitting threads.\n'
    exit()
        
