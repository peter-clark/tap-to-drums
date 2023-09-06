import numpy as np
import descriptors as desc
import flatten as flatten
import time
import socket
import re

def parse(line):
    line = str(line)
    regex = r"[-+]?\d*\.\d+|\d" # searches for all floats or integers
    list = re.findall(regex, line)
    output = [float(x) for x in list]
    return output

def parse_8(pattern):
    for step in range(len(pattern)):
        for note in range(len(pattern[step])):
            pattern[step][note] = get_eight_channel_note(int(pattern[step][note]))
    return pattern

def get_eight_channel_note(note):
    return desc.GM_dict[int(note)][5] if note!=0 else 0

def UDP_init():
        # UDP Definition
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1337
    sockt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet and UDP
    #print("------------------------------")
    #print("Connecting to "+UDP_IP+":"+str(UDP_PORT))
    #print("")
    return sockt

def EuclideanDistance(a, b):
    d = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return d

def send_poly_patt(eight_pattern, name):
    sockt=UDP_init()
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1337
    for step in range(len(eight_pattern)):
        for note in range(len(eight_pattern[step])):
            data = (str(step)+" "+str(eight_pattern[step][note])+" 1 "+name)
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            time.sleep(0.01)
    sockt.close()


def send_poly_patt_predicted(eight_pattern, name):
    sockt=UDP_init()
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1337
    for step in range(len(eight_pattern)):
        for note in range(len(eight_pattern[step])):
            data = (str(step)+" "+str(eight_pattern[step][note]*2)+" 1 "+name)
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            time.sleep(0.01)
    sockt.close()



def send_flat_patts(flatterns):
    sockt=UDP_init()
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1337
    for step in range(len(flatterns[0])):
            data = (str(step)+" 1 "+str(flatterns[1][step])+" -") # Discrete 1
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            data = (str(step)+" 2 "+str(flatterns[3][step])+" -") # Discrete 2
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            data = (str(step)+" 5 "+str(flatterns[0][step])+" -") # Continuous 1
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            data = (str(step)+" 6 "+str(flatterns[2][step])+" -") # Continuous 2
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            data = (str(step)+" 9 1 -") # all 16 [x]
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))