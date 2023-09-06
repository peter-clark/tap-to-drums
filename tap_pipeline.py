import os
import sys
import numpy as np
from natsort import natsorted
import functions as fun
import descriptors as desc
import flatten as flatten
import neuralnetwork as NN
import to_pd as puredata
import pandas as pd
import time
import socket
import re

## PATTERN INIT
# Parse all MIDI patterns in folders
all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments")

## PATTERN SELECTION
pttrn = 0
# Prompt to select pattern
pttrn = input("Select pattern index [0-1512]: ")
patt_name = all_names[int(pttrn)]
input_patt = all_pattlists[int(pttrn)]
# Flatten pattern
input_flat = flatten.flat_from_patt(input_patt)
flatterns = [[] for y in range(4)]
for i in range(len(flatterns)):
        flatterns[i] = input_flat[i]

## PUREDATA INIT
# Parse and send selected pattern
input_patt = puredata.parse_8(input_patt)

## INITIALIZE MODEL FOR PREDICTION
model_dir = os.getcwd()+"/models/continuous1.pt"
model = NN.build_model()
model.load_state_dict(NN.torch.load(model_dir))

## LOAD EMBEDDING POSITIONS
coords_dir = os.getcwd()+"/embeddings/mds.csv"
c = pd.read_csv(coords_dir)
pos = [c.X, c.Y]

## LOAD TRIANGLES, HASH_TRIANGLES, HASH_DENSITY

hash_density = 2 # number of col, row in space
 

## INITIALIZE SOCKETS FROM PUREDATA
UDP_IP_RECEIVE = "127.0.0.1"
UDP_PORT_RECEIVE = 1338
socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', UDP_PORT_RECEIVE)
socket_receive.bind(server_address)

UDP_PORT_RECEIVE_ONOFF = 1339
socket_receive_onoff = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address_onoff = ('localhost', 1339)
socket_receive_onoff.bind(server_address_onoff)

socket_receive_onoff.listen(1)
socket_receive.listen(1) # listen for tapped patterns

tappern = [0.0 for x in range(16)] # Tapped Pattern
onoff=1
while True:
    connection_onoff, client_address_onoff = socket_receive_onoff.accept()

    connection, client_address = socket_receive.accept()
    try: 
        ## TAP -> PATTERN
        # listen for taps at step
        
        """ 
        # Open connection to receive on off instructions for listening to tapped pattern
        onoff = connection_onoff.recv(8) # small buffer for on/off message, can maybe make smaller
        onoff = onoff.decode('utf-8') # text decoding
        onoff = onoff.replace('\n', '').replace('\t','').replace('\r','').replace(';','') # string cleaner just in case
        predict_tappern = False
 """
        # if tap listen is triggered in PD
        if int(onoff)==1:
            socket_receive.listen(1) # listen for tapped patterns
            connection, client_address = socket_receive.accept()
            predict_tappern = True
            while int(onoff)==1:
                # listen for on off
                onoff = connection_onoff.recv(8) # small buffer for on/off message, can maybe make smaller
                onoff = onoff.decode('utf-8') # text decoding
                onoff = onoff.replace('\n', '').replace('\t','').replace('\r','').replace(';','')

                # listen for taps at step
                tap = connection.recv(8)
                tap = tap.decode('utf-8')
                tap = tap.replace('\n', '').replace('\t','').replace('\r','').replace(';','')
                tap = tap.split(' ')
                if tap!='':
                    tappern[int(tap[1])]=float(tap[0])/127 # normalize MIDI and save to velocity array
                if tap=='':
                     onoff=0

            # if msg = '' break, as it means connection severed

        ### Once pattern is recorded in puredata, and the final array
        ### and begin prediction pipeline.
        if predict_tappern:
            print(tappern)
            ## PATTERN -> COORDINATE PREDICTION
            # Predict coordinates with model from tappern
            pred_coords = model(NN.torch.Tensor(tappern).float()).detach().numpy()
            # Return coordinates 

            ## COORDINATES -> PREDICTED PATTERN
            # Load coords, all_pattlists, embed_pos, tri, hash_tri, hash_den
            # Make pattern from predicted coords / delaunay interpol
            output_patt = fun.position2pattern(pred_coords, all_pattlists,  pos, triangles, hashed_triangles, hash_density)
            # Return output pattern

            ## PREDICTED PATTERN -> PUREDATA
            # Parse patterns into puredata send format
            puredata.send_poly_patt(input_patt,patt_name)
            output_patt = puredata.parse_8(output_patt)
            # Send predicted pattern to puredata
            puredata.send_poly_patt_predicted(output_patt, patt_name)
            # Send flattened patterns to puredata (calculated earlier)
            puredata.send_flat_patts(flatterns)
            predict_tappern = False
    finally:
         connection_onoff.close()
## LISTEN FOR NEW TAPPED PATTERN [?] (should clear pred_pattern in PD as well)