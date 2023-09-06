import os
import sys
import numpy as np
from natsort import natsorted
import functions as fun
import descriptors as desc
import flatten as flatten
import neuralnetwork as NN
import to_pd as puredata
import test_functions as tf
import pandas as pd
import time
import csv
import pickle
import socket
import re
import datetime
import pytz

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

###----------------------------------------------------------------###
## Empty array for tapped output to send to prediction
tappern = [0.0 for x in range(16)] # Tapped Pattern

## Empty array for saving results of testtapping
test_results = [[0 for x in range(32)] for y in range(3)] # results of three tap trials (consistency)
save_test_results = [0.0 for x in range((3+(32)))] # save line for csv (date time test# + test results)
results_file = os.getcwd()+"/results/taptest.csv"

## Empty array for results of save
save_line = [0.0 for x in range(86)]
tap_file = os.getcwd()+"/results/tapexplore.csv"

## OSC IP / PORT
IP = '127.0.0.1'
SEND_PORT = 1338
RECEIVE_PORT = 1339
_quit=[False]
_predict=False
_save=False
_next_pattern = False

## INITIALIZE MODEL FOR PREDICTION
model_dir = os.getcwd()+"/models/continuous1.pt"
model = NN.build_model()
model.load_state_dict(NN.torch.load(model_dir))

## PATTERN INIT + SELECTION
pickle_dir = os.getcwd()+"/data/"

## LOAD DESCRIPTORS & EMBEDDING POSITIONS
descriptor_file = open(pickle_dir+"descriptors.pkl","rb")
d = pickle.load(descriptor_file)
descriptor_file.close()

mds_pos_file = open(pickle_dir+"mds_pos.pkl", 'rb')
mds_pos = pickle.load(mds_pos_file) 
mds_pos_file.close()

# Load patterns from pickle file
patt_file = open(pickle_dir+"patterns.pkl", 'rb')
all_pattlists = pickle.load(patt_file)
patt_file.close()

# Load names from pickle file
name_file = open(pickle_dir+"pattern_names.pkl","rb")
all_names = pickle.load(name_file)
name_file.close()

sections=[[]for x in range(16)] 
names = [[] for x in range(16)]
pattern_idxs = [[]for x in range(16)]
tf.make_pos_grid(mds_pos,sections,pattern_idxs,all_names,names, False) # bool is show plot
# tf.print_sections(sections)



# Select pattern from sections
test_patterns = [0 for x in range(len(pattern_idxs)+2)]
""" FOR GETTING PATTERNS FROM A SECTION
t_i=0
for i in range(len(pattern_idxs[11])):
    if sections[11][i][0]>0.8 and sections[11][i][1]<0.705:
        print(f"12: {pattern_idxs[11][i]}")
        print(f"12: {sections[11][i]}")
        if t_i == len(test_patterns):
            break
        test_patterns[t_i]=pattern_idxs[11][i]
        t_i+=1
# ADD TWO CONTROL 
i = np.random.randint(0,16)
test_patterns[16] = test_patterns[i]
j = np.random.randint(0,16)
while j==i:
    j = np.random.randint(0,16)
test_patterns[17] = test_patterns[j]
"""

test_patterns = [
    894, 423, 1367, 249, 
    939, 427, 590, 143,
    912, 678, 1355, 580,
    1043, 673, 1359, 736,
    678, 1355
    ]
np.random.shuffle(test_patterns)
for i in range(len(test_patterns)-1):
    if test_patterns[i]==test_patterns[i+1]:
        np.random.shuffle(test_patterns)
        break
print(test_patterns)

next_patt = test_patterns[0]
patt_name = all_names[int(next_patt)]
input_patt = all_pattlists[int(next_patt)]

def get_flat(pattern):
    # Flatten pattern and organize
    input_flat = flatten.flat_from_patt(input_patt)
    flatterns = [[] for y in range(4)]
    for i in range(len(flatterns)):
            flatterns[i] = input_flat[i]
    return flatterns

flatterns = get_flat(input_patt)

## LOAD TRIANGLES, HASH_TRIANGLES, HASH_DENSITY
triangles_file = open(pickle_dir+"triangles.pkl", 'rb')
triangles = pickle.load(triangles_file) 
triangles_file.close()

hash_density = 2 # number of col, row in space

hashed_triangles_file = open(pickle_dir+"hashed_triangles.pkl", 'rb')
hashed_triangles = pickle.load(hashed_triangles_file) 
hashed_triangles_file.close()

# Create instance of sender class
send_to_pd = SimpleUDPClient(IP, SEND_PORT)
# Create instance of receiver class
dispatcher = Dispatcher()

## Define handlers for messages
def tap_message_handler(address, *args): # /tap
    print(address)
    if address == "/tap/predict":
        global _predict
        _predict=True
        print("Predicting...")
    for idx in range(len(args)):
        tappern[idx]=(args[idx]/127) if args[idx]>=0.0 else 0.0
    #print(f"Tapped Pattern: {tappern}")

def save_data_message_handler(address, *args): # save information from puredata
    if address=="/save":
        # 1. DATE
        today = datetime.datetime.now(pytz.timezone("Europe/Madrid"))
        date = today.date()
        save_line[0] = date
        # 2. TIME
        time = today.strftime("%H:%M:%S")
        save_line[1] = time
        # 3. INPUT PATT
        save_line[2] = next_patt
        # 4. INPUT PATT NAME
        save_line[3] = patt_name
        # 5-20. Tapped Pattern [x16]
        for idx in range(len(tappern)):
            save_line[4+idx] = tappern[idx]
            save_line[20+idx] = flatterns[0][idx] # c1
            save_line[36+idx] = flatterns[1][idx] # d1
            save_line[52+idx] = flatterns[2][idx] # c2
            save_line[68+idx] = flatterns[3][idx] # d2
    if address=="/save/bpm":
        save_line[85]=args[0]
    if address=="/save":
        with open(tap_file,'a') as results: 
                wr = csv.writer(results)
                wr.writerow(save_line)
                results.close()

def test_results_message_handler(address, *args): # /get_prediction (bool)
    today = datetime.datetime.now(pytz.timezone("Europe/Madrid"))
    date = today.date()
    save_test_results[0]=date
    time = today.strftime("%H:%M:%S")
    save_test_results[1]=time
    if(address=="/test_results/testnum"):
        save_test_results[2]=int(args[0])
        #print(f"args {args[0]}")
    elif(save_test_results[2]>0):
        for i in range(len(args)):
            save_test_results[i+3]=args[i]
        with open(results_file,'a') as results: 
            wr = csv.writer(results)
            wr.writerow(save_test_results)
            results.close()
    if(save_test_results[2]==3):
        print("Saved Tap Consistency Test Results to CSV.")

def test_message_handler(address, *args):
    if address == "/test/next_pattern":
        global _next_pattern 
        _next_pattern = True
        print("Going next pattern...")


def joystick_message_handler(address, *args): # /joystick (xy)
    print("yadda yadda.")


def quit_message_handler(address, * args): # /quit
    _quit[0]=True
    print("I'm out.")



# Pass handlers to dispatcher
dispatcher.map("/tap*", tap_message_handler)
dispatcher.map("/save*", save_data_message_handler)
dispatcher.map("/test_results*", test_results_message_handler)
dispatcher.map("/test*", test_message_handler)
dispatcher.map("/joystick*", joystick_message_handler)
dispatcher.map("/quit*", quit_message_handler)

# Define default handler
def default_handler(address, *args):
    print(f"Nothing done for {address}: {args}")
dispatcher.set_default_handler(default_handler)

# Establish UDP connection with PureData
server = BlockingOSCUDPServer((IP, RECEIVE_PORT), dispatcher)

###----------------------------------------------------------------###
def pattern_to_pd(pattern, name, udp, type=0):
    ## Type 0 --> Input Pattern (default)
    ## Type 1 --> Predicted Pattern (midi channels *2 for PureData)
    ## Type 2 --> Flattened Patterns (different channel routing)
    if type==0:
        for step in range(len(pattern)):
            for note in range(len(pattern[step])):
                udp.send_message("/pattern/channel",pattern[step][note])
                udp.send_message("/pattern/step",step)
                udp.send_message("/pattern/velocity",1)
        print(f"Sent pattern: {patt_name}")
        #print(f"Sent pattern: {pattern}")
    if type==1:
        for step in range(len(pattern)):
            for note in range(len(pattern[step])):
                if pattern[step][note]!=0:
                    n = desc.GM_dict[int(pattern[step][note])][5]
                    udp.send_message("/pattern/channel",(n*2))
                    udp.send_message("/pattern/step",step)
                    udp.send_message("/pattern/velocity",1)
        print("Sent predicted pattern.")
    if type==2:
        #d1->1, d2->2, c1->5, c2->6, all->9
        for alg in range(4):     
            for step in range(len(pattern[0])):
                channel=9
                if alg==0: #c1
                    channel=5
                elif alg==1: #d1
                    channel=1
                elif alg==2: #c2
                    channel=6
                elif alg==3: #d2
                    channel=2
                udp.send_message("/pattern/channel",channel)
                udp.send_message("/pattern/step",step)
                udp.send_message("/pattern/velocity",pattern[alg][step])
        print("Sent flattened patterns.")
###



## 
""" 
What about bias in the sense that participants can see the drum patterns, hinting at implied rhythm?
"""
##

current_test = 0
while _quit[0] is False:
    server.handle_request()
    if _predict:
        pred_coords = model(NN.torch.Tensor(tappern).float()).detach().numpy()
        output_patt = fun.position2pattern(pred_coords, all_pattlists,  mds_pos, triangles, hashed_triangles, hash_density)
        print(f"Predicted Pattern: {output_patt}")
        pattern_to_pd(output_patt, patt_name, send_to_pd, type=1)
        _predict=False

    if _next_pattern:
        # send next pattern in list to puredata
        if(current_test>len(test_patterns)-1):
            print("Testing is over. \n \nThank you for participating.")
            _next_pattern = False
            break
        next_patt = int(test_patterns[current_test])
        input_patt = all_pattlists[next_patt]
        patt_name = all_names[next_patt]
        flatterns = get_flat(input_patt)
        input_patt = puredata.parse_8(input_patt) # edit this to be done in send function, not separate file
        pattern_to_pd(input_patt, patt_name, send_to_pd, type=0)
        print(f"Current test [{current_test}]: {test_patterns[current_test]}")
        current_test+=1
        _next_pattern = False