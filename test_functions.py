import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def EuclideanDistance(a, b):
    d = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return d

def get_random_selection():
    print()

def make_pos_grid(data, sections, indices, all_names,names, _plot=False):
    '''
    Takes positions, and returns patterns 16 sections.                                                                                                                                                         
[12][13][14][15]
[08][09][10][11]
[04][05][06][07]
[00][01][02][03]
    '''
    x = [pos[0] for pos in data[:]]
    y = [pos[1] for pos in data[:]]
        
    for i in range(len(x)):
        if y[i]<=0.25: # row 1 --------------------------------------------------------
            if x[i]<=0.25: # col 1
                sections[0].append([x[i],y[i]])
                indices[0].append(i)
                names[0].append(all_names[i])
            elif x[i]<=0.50 and x[i]>0.25: # col 2
                sections[1].append([x[i],y[i]])
                indices[1].append(i)
                names[1].append(all_names[i])
            elif x[i]<=0.75 and x[i]>0.5:
                sections[2].append([x[i],y[i]])
                indices[2].append(i)
                names[2].append(all_names[i])
            else:
                sections[3].append([x[i],y[i]])
                indices[3].append(i)
                names[3].append(all_names[i])
        elif y[i]<=0.5: # row 2 --------------------------------------------------------
            if x[i]<=0.25: # col 1
                sections[4].append([x[i],y[i]])
                indices[4].append(i)
                names[4].append(all_names[i])
            elif x[i]<=0.50 and x[i]>0.25:
                sections[5].append([x[i],y[i]])
                indices[5].append(i)
                names[5].append(all_names[i])
            elif x[i]<=0.75 and x[i]>0.5:
                sections[6].append([x[i],y[i]])
                indices[6].append(i)
                names[6].append(all_names[i])
            else:
                sections[7].append([x[i],y[i]])
                indices[7].append(i)
                names[7].append(all_names[i])
        elif y[i]<=0.75: # row 3 --------------------------------------------------------
            if x[i]<=0.25: # col 1
                sections[8].append([x[i],y[i]])
                indices[8].append(i)
                names[8].append(all_names[i])
            elif x[i]<=0.50 and x[i]>0.25:
                sections[9].append([x[i],y[i]])
                indices[9].append(i)
                names[9].append(all_names[i])
            elif x[i]<=0.75 and x[i]>0.5:
                sections[10].append([x[i],y[i]])
                indices[10].append(i)
                names[10].append(all_names[i])
            else:
                sections[11].append([x[i],y[i]])
                indices[11].append(i)
                names[11].append(all_names[i])
        else: # row 4 ----------------------------------------------------------------------
            if x[i]<=0.25: # col 1
                sections[12].append([x[i],y[i]])
                indices[12].append(i)
                names[12].append(all_names[i])
            elif x[i]<=0.50 and x[i]>0.25:
                sections[13].append([x[i],y[i]])
                indices[13].append(i)
                names[13].append(all_names[i])
            elif x[i]<=0.75 and x[i]>0.5:
                sections[14].append([x[i],y[i]])
                indices[14].append(i)
                names[14].append(all_names[i])
            else:
                sections[15].append([x[i],y[i]])
                indices[15].append(i)
                names[15].append(all_names[i])
    if _plot:
        plt.scatter(x,y, c='lightskyblue',s=7)
        _x=[]
        _y=[]
        for i in range(len(x)):
            if i==894 or i==423 or i==1367 or i==249 or i==939 or i==427 or i==590 or i==143 or i==912 or i==678 or i==1355 or i==580 or i==1043 or i==673 or i==1359 or i==736:
                _x.append(x[i])
                _y.append(y[i])
                plt.text(x[i],y[i],str(i))
        plt.scatter(_x, _y, c='orangered',s=12, marker='x')
        plt.title("Polyphonic Rhythm Space and Selected Patterns for Testing", fontfamily='serif')
        """
        #boi
        plt.plot((1,1), (0,1), c='grey', linewidth=0.75, linestyle='--') #right
        plt.plot((1,0), (0,0), c='grey', linewidth=0.75, linestyle='--') #bottom
        plt.plot((1,0), (1,1), c='grey', linewidth=0.75, linestyle='--') #top
        plt.plot((0,0), (1,0), c='grey', linewidth=0.75, linestyle='--') #left
        #verts
        plt.plot((0.25,0.25), (0,1), c='grey', linewidth=0.75, linestyle='--') #25%
        plt.plot((0.5,0.5), (0,1), c='grey', linewidth=0.75, linestyle='--') #50%
        plt.plot((0.75,0.75), (0,1), c='grey', linewidth=0.75, linestyle='--') #75%
        #horiz
        plt.plot((0,1),(0.25,0.25), c='grey', linewidth=0.75, linestyle='--') #25%
        plt.plot((0,1),(0.5,0.5), c='grey', linewidth=0.75, linestyle='--') #50%
        plt.plot((0,1),(0.75,0.75), c='grey', linewidth=0.75, linestyle='--') #75%
        """
        #plt.plot((x_min,y_min),(x_min,0), c='grey')
        #plt.plot((x_min,y_min),(x_max,0), c='grey')

        plt.show()

def print_sections(sections):
    n=0
    t=0
    for i in range(16):
        print(f"[{t},{i%4}]: {len(sections[i])}")
        n += len(sections[i])
        if (i+1)%4==0 and i!=0:
            t+=1
    print(f"Total: {n}")
