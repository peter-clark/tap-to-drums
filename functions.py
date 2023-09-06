# functions for making rhythmspaces

import os
import mido
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import math

GM_dict={
# key is midi note number
# values are:
# [0] name (as string)
# [1] name category low mid or high (as string)
# [2] substiture midi number for simplified MIDI (all instruments)
# [3] name of instrument for 8 note conversion (as string)
# [4] number of instrument for 8 note conversion 
# [5] substiture midi number for conversion to 8 note
# [6] substiture midi number for conversion to 16 note
# [7] substiture midi number for conversion to 3 note
# if we are going to remap just use GM_dict[msg.note][X]

    22:['Closed Hi-Hat edge', 'high', 42, 'CH', 3,42,42,42],
    26:['Open Hi-Hat edge', 'high', 46, 'OH', 4,46,46,42],
    35:['Acoustic Bass Drum','low',36, 'K', 1, 36,36,36],
    36:['Bass Drum 1','low',36, 'K', 1, 36,36,36],
    37:['Side Stick','mid',37, 'RS', 6, 37,37,38],
    38:['Acoustic Snare','mid',38, 'SN', 2, 38,38,38],
    39:['Hand Clap','mid',39, 'CP', 5, 39, 39,38],
    40:['Electric Snare','mid',38, 'SN', 2, 38,38,38],
    41:['Low Floor Tom','low',45, 'LT', 7, 45,45,36],
    42:['Closed Hi Hat','high',42, 'CH', 3, 42,42,42],
    43:['High Floor Tom','mid',45, 'HT', 8, 45,45,38],
    44:['Pedal Hi-Hat','high',46, 'OH', 4, 46, 46,42],
    45:['Low Tom','low',45, 'LT', 7, 45, 45,36],
    46:['Open Hi-Hat','high',46, 'OH', 4, 46, 46,42],
    47:['Low-Mid Tom','low',47, 'MT', 7, 45, 47,36],
    48:['Hi-Mid Tom','mid',47, 'MT', 7, 50, 50,38],
    49:['Crash Cymbal 1','high',49, 'CC', 4, 46, 42,42],
    50:['High Tom','mid',50, 'HT', 8, 50, 50,38],
    51:['Ride Cymbal 1','high',51, 'RC', -1, 42, 51,42],
    52:['Chinese Cymbal','high',52, '', -1, 46, 51,42],
    53:['Ride Bell','high',53, '', -1, 42, 51,42],
    54:['Tambourine','high',54, '', -1, 42, 69,42],
    55:['Splash Cymbal','high',55, 'OH', 4, 46, 42,42],
    56:['Cowbell','high',56, 'CB', -1, 37, 56,42],
    57:['Crash Cymbal 2','high',57,'CC', 4,46, 42,42],
    58:['Vibraslap',"mid",58,'VS', 6,37, 37,42],
    59:['Ride Cymbal 2','high',59, 'RC',3, 42, 51,42],
    60:['Hi Bongo','high',60, 'LB', 8, 45,63,42],
    61:['Low Bongo','mid',61, 'HB', 7, 45, 64,38],
    62:['Mute Hi Conga','mid',62, 'MC', 8, 50, 62,38],
    63:['Open Hi Conga','high',63, 'HC', 8, 50, 63,42],
    64:['Low Conga','low',64, 'LC', 7, 45,64,36],
    65:['High Timbale','mid',65, '',8, 45,63,38],
    66:['Low Timbale','low',66, '',7, 45,64,36],
    67:['High Agogo','high',67, '',-1, 37,56,42],
    68:['Low Agogo','mid',68,'',- 1 , 37,56,38],
    69:['Cabasa','high',69, 'MA',-1, 42,69,42],
    70:['Maracas','high',69, 'MA',-1, 42,69,42],
    71:['Short Whistle','high',71,'',-1,37, 56,42],
    72:['Long Whistle','high',72,'',-1,37, 56,42],
    73:['Short Guiro','high',73,'',-1, 42,42,42],
    74:['Long Guiro','high',74,'',-1,46,46,42],
    75:['Claves','high',75,'',-1, 37,75,42],
    76:['Hi Wood Block','high',76,'',8, 50,63,42],
    77:['Low Wood Block','mid',77,'',7,45, 64,38],
    78:['Mute Cuica','high',78,'',-1, 50,62,42],
    79:['Open Cuica','high',79,'',-1, 45,63,42],
    80:['Mute Triangle','high',80,'',-1, 37,75,42],
    81:['Open Triangle','high',81,'',-1, 37,75,42],
    }

def midipattern2pattlist(pattern_name, instruments):
# pattern name must include .mid
# get the pattern and convert it to an on/off grid
# use the "instruments" variable to define instrument mapping "all", "16", "8", "3"
	pattern=[]
	mid=mido.MidiFile(pattern_name) #create a mido file instance
	#print("midi ticks per beat ", mid.ticks_per_beat)
	sixteenth= mid.ticks_per_beat/4 #find the length of a sixteenth note
	#print ("sixteenth", sixteenth)

	# time: inside a track, it is delta time in ticks (integrer). 
 	# A delta time is how long to wait before the next message.
	accumulated=0 #use this to keep track of time 

	# depending on the instruments variable select a notemapping

	if instruments=="allinstruments":
		column=2
	elif instruments=="16instruments":
		column=6
	elif instruments=="8instruments":
		column=5
	elif instruments=="3instruments":
		column=7

	for i, track in enumerate(mid.tracks):
		for msg in track:
			# print(msg)
			if msg.type == "note_on":
				#quantize notes > 0.5 to the next step
				#print("time", msg.time, "note", msg.note, "quantized", accumulated//sixteenth)
				quantized_time=int((msg.time+((sixteenth*0.45)/sixteenth))//1)
				accumulated+=quantized_time #adds if note comes after a note off and a silence
 				#print("accumulated", accumulated)
 				#print (int(accumulated/sixteenth), msg.note)

 				#remap msg.note by demand
 				#print ("note", msg.note)
				midinote = GM_dict[msg.note][column]

				pattern.append((int(accumulated/sixteenth),midinote)) # step, note
			elif msg.type=="note_off":
				accumulated+=msg.time #adds noteoff time
 				#print("accumulated", accumulated)
			elif msg.type == "control_change":
				accumulated+=msg.time #adds cc time
 				#print("accumulated", accumulated)
		if len(pattern)>0: #just proceed if analyzed pattern has at least one onset

 			#round the pattern to the next multiple of 16
			pattern_lenth_in_steps=(int(accumulated//sixteenth//16)+int((accumulated//sixteenth%16)+15)//16)*16
 			#create an empty list of lists the size of the pattern
			output_pattern=[[]]*pattern_lenth_in_steps 
 			# group the instruments that played at a specific step
			for step in range(len(output_pattern)):
				output_pattern[step]=([x[1] for x in pattern if x[0]==step])

 				#make sure no notes are repeated and events are sorted
				output_pattern[step] = list(set(output_pattern[step]))
				output_pattern[step].sort()

 			# print (output_pattern)
 	##################################
 	# split the pattern every 16 steps
 	##################################
	pattlist_split=[]
	for x in range(len(output_pattern)//16):
		patt_fragment = output_pattern[x*16:(x*16)+16]
		patt_density = sum([1 for x in patt_fragment if x!=[]])
		
		#############################################################
		# filter out patterns that have less than 4 events with notes
		#############################################################
		# NOTE: more conditions could be added (i.e. kick on step 0)
		############################################################# 
		if patt_density > 4:
			#print(pattern_name, x, sum([1 for x in patt_fragment if x!=[]]))
			pattlist_split.append(patt_fragment)
	
	###################################
	# output a list of grids 
	# containing the differnet patterns 
	# of 16 steps
	# found in each midi file
	###################################

	return pattlist_split

def folder2filelist(foldername):
# get a folder name and list all midi files
	allfiles=os.listdir(foldername)
	#while '.DS_Store' in allfiles: allfiles.remove('.DS_Store') #remove "DS_Store" file	
	allfiles=[foldername+"/"+x for x in allfiles]
	allfiles=[x for x in allfiles if os.path.isfile(x)==True and x[-4:]==".mid"]
	allfiles.sort()
	return allfiles

def make_iterative_names(filename, iterations):
#create iterative names for a string
	new_pattlist_names=[]
	pattlist_16steps_name= filename.split("/")[-1] #remove folder names
	pattlist_16steps_name= pattlist_16steps_name.split(".")[0] #remove .mid extension
	#print("filename", pattlist_16steps_name)
	if iterations>1:
		for i,iteration in enumerate (range(iterations)): #enumerate

			#pattlist_16steps_name=pattlist_16steps_name+"_"+str(i)
			#print ("name",i, pattlist_16steps_name+"_"+str(i))
			new_pattlist_names.append(pattlist_16steps_name+"_"+str(i))
	
	else:

		new_pattlist_names.append(pattlist_16steps_name)
	
	return new_pattlist_names

def folder2pattlist (foldername, instruments):
# a "pattlist" is a specific data structure used to store midi onsets,
# therefore very useful to represent symbolic drum patterns.
# This function starts with a folder name to look for midi files inside it,
# then converts all of them to 16 step (fixed in midipattern2pattlist())
# pattlists and creates a specific name for each subpattern (filename_n)
# being "n" the nth subpattern within the midi file (this is done in
# make_iterative_names()).
	
	filenames=folder2filelist(foldername)
	#print (filenames)

	all_names=[]
	all_pattlists=[]

	for filename in filenames:
		#print(filename)
		list_of_pattlists=midipattern2pattlist(filename, instruments) #convert a midi file into a list of pattlists
		for pattlist in list_of_pattlists:
			all_pattlists.append(pattlist)
			#print ("pattlist", pattlist)
		new_pattlist_names=make_iterative_names(filename, len(list_of_pattlists)) #make a list of filenames, useful if file contains more than one grid
		for name in new_pattlist_names:
			#print("name", name)

			all_names.append(name)
		
	#print (all_names)
	#print("len", len(all_pattlists))
	#print (all_pattlists)

	return all_pattlists, all_names

def rootfolder2pattlists (foldername, instruments):
# process all midi files in a root folder by  
# converting them to pattlists and splitting in 16 steps
	all_pattlists = []
	all_names = []
	for root, dirs, files in os.walk("midi/"):
		#print ("root", root)
		pattlists, names = folder2pattlist(root, "allinstruments")
		#print ("names", names)
		all_pattlists += pattlists
		all_names += names
		#print (all_names)
	return all_pattlists, all_names

def d_mtx_2_mds(d_mtx):
# input the descriptor matrix
#  copmute MDS
# output 2D positions of each pattern

	d_mtx = d_mtx - d_mtx.min(axis=0) #normalize by max
	d_mtx = d_mtx / d_mtx.max(axis=0) #normalize by max
	d_mtx[np.isnan(d_mtx)] = 0 # avoid nans, convert to 0 (i.e. a row only has 0/0)

	# pairwise comparison among patterns (creates a dissimilarity matrix)
	#pw=pairwise_distances(d_mtx, Y=None, metric='euclidean')
	
	# instantiate MDS and get positions
	seed = np.random.RandomState(seed=3) #parameters of the MDS
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, n_jobs=1)# dissimilarity="precomputed", )
	#pos = mds.fit(pw).embedding_ #get positions using MDS
	pos = mds.fit(d_mtx).embedding_ #get positions using MDS
	clf = PCA(n_components=2) #setup PCA
	pos = clf.fit_transform(pos) #Rotate points using PCA to achieve better alignment

	pos = pos - np.min(pos)
	pos = pos / np.max(pos)

	return pos

def pointer2hash (s, rows_cols):
# input a 2D pointer s and output the hash it belongs to
	out_h = (rows_cols**2)
	for i,h in enumerate(range(rows_cols**2)):
		x_bound_1 = (h%rows_cols)/rows_cols
		x_bound_2 = ((h%rows_cols) + 1)/rows_cols
		y_bound_1 = (h//rows_cols)/rows_cols
		y_bound_2 = ((h//rows_cols) + 1)/rows_cols

		if s[0] > x_bound_1 and s[0] <= x_bound_2 and s[1] > y_bound_1 and s[1] <= y_bound_2:
			out_h = i
	
	return out_h

def hash_triangles(pos, triangles, rows_cols):
# input all triangles and hash them according to their position
# the number of hashes is (rows_cols**2)+1
# because some triangles don't have all points in the same hash

	dictionary={}
	# prepare dictionary with the right amount of empty lists
	for n in range((rows_cols**2)+1): # +1 because there are sme patterns on the edge of the divisions
		dictionary[n]=[]
		row=n/rows_cols
		n=n+rows_cols
		col=n%rows_cols

	# store triangles in one of the categories
	for t, triangle in enumerate (triangles):
		triangle_quadrant_list = []
		for point in triangle:
			point_coord=pos[point]
			#print ("coord", point_coord)
			pointhash = pointer2hash(point_coord,rows_cols)
			#print ("pointhash", pointhash)
			triangle_quadrant_list.append(pointhash)
		#print ("tql", triangle_quadrant_list)
		if len(set(triangle_quadrant_list)) == 1: # all vertices in same quadrant

			dictionary[triangle_quadrant_list[0]].append(t)
		else:
			dictionary[(rows_cols**2)].append(t)
		#print (dictionary)
	return dictionary

	
	return dictionary

def create_delaunay_triangles (point_coordenates):
#create delaunay triangles from a list of (x,y) points
	points=np.array(point_coordenates)
	pointsincols=[]

	# Nomralize the coordinates and save in N_points
	for i in range(len(points[0])):
		N=[x[i] for x in points]
		N=np.array(N)
		N=(N-min(N))/(max(N)-min(N))
		pointsincols.append(N)
	N_points=[]
	for row in range(len(pointsincols[0])):
		N_point=[x[row] for x in pointsincols]
		N_points.append(N_point)
	N_points=np.array(N_points)


	#generate triangles
	tri = Delaunay(N_points)
	listoftri=tri.simplices
	#print tri.simplices
	for t in listoftri:
		tripoints=N_points[t[0]][0],N_points[t[0]][1], N_points[t[1]][0],N_points[t[1]][1], N_points[t[2]][0],N_points[t[2]][1]
		
		#print (tripoints) #these are the coordinates of the three points of each triangle

	#returns the list of triangles of the space.
	return listoftri

def triangarea(p1,p2,p3):
# input the coordinates of three vertices of a triangle and output its area
	area=abs(0.5 * (((p2[0]-p1[0])*(p3[1]-p1[1]))-((p3[0]-p1[0])*(p2[1]-p1[1]))))
	return area

def isinside(s,pos, tri):
# input the positions of the pointer "s" and the triangle vertexes tri[0,1,2]
# report 1 or 0 if s is inside or outside tri
	a1=triangarea(pos[tri[0]],pos[tri[1]],pos[tri[2]])
	a2=triangarea(pos[tri[0]],pos[tri[1]],s)
	a3=triangarea(pos[tri[1]],pos[tri[2]],s)
	a4=triangarea(pos[tri[0]],pos[tri[2]],s)
	if np.abs(a1-(a2+a3+a4))<0.000001: # Low value to fix float issues
		inside = 1
	else:
		inside = 0
	return inside

def find_intersection( p0, p1, p2, p3 ) :
# find intersection point between two lines
# input two points for each line
# output intrsection point

    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]
    
    denom = s10_x * s32_y - s32_x * s10_y

    #if denom == 0 : return None # collinear

    denom_is_positive = denom > 0

    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]

    s_numer = s10_x * s02_y - s10_y * s02_x

    #if (s_numer < 0) == denom_is_positive : return None # no collision

    t_numer = s32_x * s02_y - s32_y * s02_x

    #if (t_numer < 0) == denom_is_positive : return None # no collision

    #if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : return None # no collision


    # collision detected

    t = t_numer / denom

    intersection_point = [ p0[0] + (t * s10_x), p0[1] + (t * s10_y) ]


    return intersection_point

def distance(a,b):
	d=math.sqrt(pow(abs(a[0]-b[0]),2)+pow(abs(a[1]-b[1]),2))
	return d

def weights(s,p1,p2,p3):
# input a pointer s and the three vertexes of a triangle
# output the CLOSENESS (not distance!) of the pointer with each vertex
	w1=1-(float(distance(s,p1))/distance(p1,find_intersection(s,p1,p2,p3)))
	w2=1-(float(distance(s,p2))/distance(p2,find_intersection(s,p2,p1,p3)))
	w3=1-(float(distance(s,p3))/distance(p3,find_intersection(s,p3,p1,p2)))
	return w1,w2,w3

def search_in_space(search_coordinates, triangles, coordinates, quadrant_2_triangles, quadrant):
# input 2D normalized coordinates and search to which triangle it belongs to,
# compute the weights of each vertex of the triangle towards the pointer
# output the three weights
	
	##############################################################
	# iterate over all triangles and see if search_coordinates are inside
	listtestinside=[]
	triweight=[0,0,0]
	############
	boundingtriangle=0
	sublist_of_triangles=quadrant_2_triangles[quadrant]+quadrant_2_triangles[len(quadrant_2_triangles.keys())-1]
	n=0
	while boundingtriangle == 0 and n<len(sublist_of_triangles):
		#sublist_of_triangles=quadrant_2_triangles[quadrant]+quadrant_2_triangles[5]
		#print "sublist_of_triangles", sublist_of_triangles
		#tri=triangles[n]
		tri=triangles[sublist_of_triangles[n]]

		boundingtriangle=isinside(search_coordinates, coordinates, tri) #look for bounding triangle
		if boundingtriangle==1:
			#intersec1= find_intersection(search_coordinates,coordinates[triangles[n][0]],coordinates[triangles[n][1]],coordinates[triangles[n][2]])
			intersec1= find_intersection(search_coordinates,coordinates[triangles[sublist_of_triangles[n]][0]],coordinates[triangles[sublist_of_triangles[n]][1]],coordinates[triangles[sublist_of_triangles[n]][2]])
			#w= weights (search_coordinates, coordinates[triangles[n][0]], coordinates[triangles[n][1]], coordinates[triangles[n][2]])
			w= weights (search_coordinates, coordinates[triangles[sublist_of_triangles[n]][0]], coordinates[triangles[sublist_of_triangles[n]][1]], coordinates[triangles[sublist_of_triangles[n]][2]])
			w=np.array(w)
			w=w/float(sum(w))
			#triweight=float(triangles[n][0]),w[0],float(triangles[n][1]),w[1],float(triangles[n][2]),w[2]
			triweight=float(triangles[sublist_of_triangles[n]][0]),w[0],float(triangles[sublist_of_triangles[n]][1]),w[1],float(triangles[sublist_of_triangles[n]][2]),w[2]
			break
		n += 1
	###########

	return triweight

def pattern_interpolation(pattern1, w1, pattern2, w2, pattern3, w3): 
# interpolate bewteen three patterns using weight values 
# inputs are three patterns and three weights

	###############################2020###############################
	#compute density ratio
	###############################2020###############################
	# density ratio = probability of finding a specific onset within the complete pattern
	density_ratio_1={}
	all_onsets_1 = [item for sublist in pattern1 for item in sublist]
	for key in set(all_onsets_1):
		density_ratio_1[key]=len([x for x in all_onsets_1 if x==key])
	sum_all_onsets_1=sum(density_ratio_1.values()) #total number of onsets
	for key in density_ratio_1.keys():
		density_ratio_1[key]=density_ratio_1[key]/float(sum_all_onsets_1)
	#print density_ratio_1
	
	density_ratio_2={}
	all_onsets_2 = [item for sublist in pattern2 for item in sublist]
	for key in set(all_onsets_2):
		density_ratio_2[key]=len([x for x in all_onsets_2 if x==key])
	sum_all_onsets_2=sum(density_ratio_2.values()) #total number of onsets
	for key in density_ratio_2.keys():
		density_ratio_2[key]=density_ratio_2[key]/float(sum_all_onsets_2)
	#print density_ratio_2

	density_ratio_3={}
	all_onsets_3 = [item for sublist in pattern3 for item in sublist]
	for key in set(all_onsets_3):
		density_ratio_3[key]=len([x for x in all_onsets_3 if x==key])
	sum_all_onsets_3=sum(density_ratio_3.values()) #total number of onsets
	for key in density_ratio_3.keys():
		density_ratio_3[key]=density_ratio_3[key]/float(sum_all_onsets_3)
	#print density_ratio_3

	output_pattern=[]
	for i in range(len(pattern1)):

		# make a list with all the instruments of the step
		step1=pattern1[i]
		step2=pattern2[i]
		step3=pattern3[i]

		#make a list of lists, each list being one of the weights for each instrument in step1, estep2, step3
		step1_weights=[]
		step2_weights=[]
		step3_weights=[]

		###############################2020###############################
		# add pattern weight to weight list
		###############################2020###############################

		step1_weights.append([w1]*len(pattern1[i]))
		step2_weights.append([w2]*len(pattern2[i]))
		step3_weights.append([w3]*len(pattern3[i]))

		###############################2020###############################
		# compute frequency class weihgt
		###############################2020###############################
		high_class=[42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 59, 60, 63, 67, 69, 70, 71, 72, 74,75, 76, 78, 79, 80, 81]
		high_class_weight=0.03333333333333
		mid_class=[37, 38, 39, 40, 43, 50, 58, 61, 62, 65, 73,  77]
		mid_class_weight=1.33333333333
		low_class=[41, 45, 47, 48, 64, 66, 68]
		low_class_weight= 0.3
		super_low_class=[35, 36]
		super_low_class_weight= 0.53333333333333

		fcw1=[]
		fcw2=[]
		fcw3=[]

		for instrument in pattern1[i]:
			if instrument in high_class:
				fcw1.append(high_class_weight)
			else:
				if instrument in mid_class:
					fcw1.append(mid_class_weight)
				else:
					if instrument in low_class:
						fcw1.append(low_class_weight)
					else:
						if instrument in super_low_class:
							fcw1.append(super_low_class_weight)
						else:
							fcw1.append(0)
		step1_weights.append(fcw1)

		for instrument in pattern2[i]:
			if instrument in high_class:
				fcw2.append(high_class_weight)
			else:
				if instrument in mid_class:
					fcw2.append(mid_class_weight)
				else:
					if instrument in low_class:
						fcw2.append(low_class_weight)
					else:
						if instrument in super_low_class:
							fcw2.append(super_low_class_weight)
						else:
							fcw2.append(0)
		step2_weights.append(fcw2)

		for instrument in pattern3[i]:
			if instrument in high_class:
				fcw3.append(high_class_weight)
			else:
				if instrument in mid_class:
					fcw3.append(mid_class_weight)
				else:
					if instrument in low_class:
						fcw3.append(low_class_weight)
					else:
						if instrument in super_low_class:
							fcw3.append(super_low_class_weight)
						else:
							fcw3.append(0)
		step3_weights.append(fcw3)

		###############################2020###############################
		# compute syncopation value weight
		###############################2020###############################
		metricalweight=[5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
		salience=[0.19999999999999996, 1.0, 0.7999999999999999, 1.0, 0.6, 1.0, 0.7999999999999999, 1.0, 0.3999999999999999, 1.0, 0.7999999999999999, 1.0, 0.6, 1.0, 0.7999999999999999, 1.0]
		syncopation_weight1=[]
		syncopation_weight2=[]
		syncopation_weight3=[]

		for instrument in pattern1[i]:
			if instrument not in pattern1[(i+1)%12]:
				syncopation_weight1.append(salience[i])
			else:
				syncopation_weight1.append(0.01) #if the pattern is a nothing append a small weight
		for instrument in pattern2[i]:
			if instrument not in pattern2[(i+1)%12]:
				syncopation_weight2.append(salience[i])
			else:
				syncopation_weight2.append(0.01) #if the pattern is a nothing append a small weight
		for instrument in pattern3[i]:
			if instrument not in pattern3[(i+1)%12]:
				syncopation_weight3.append(salience[i])
			else:
				syncopation_weight3.append(0.01) #if the pattern is a nothing append a small weight
		step1_weights.append(syncopation_weight1)
		step2_weights.append(syncopation_weight2)
		step3_weights.append(syncopation_weight3)


		###############################2020###############################
		#compute onset pattern weight (opw)
		###############################2020###############################
		# opw is the multiplication of all weights
		
		instrument_and_weight={}

		# print "step1", step1, step1_weights
		# print "step2", step2, step2_weights
		# print "step3", step3, step3_weights
		
		# make a dictionary with all instruments as keys and 0 as initial value
		for instrument in set(step1+step2+step3):
			instrument_and_weight[instrument]=[0]

		# multiply all indexes	
		for index,instrument in enumerate(step1):
			opw=1
			for weight in step1_weights:
				opw=opw*weight[index]
			instrument_and_weight[instrument].append(opw)

		for index,instrument in enumerate(step2):
			opw=1
			for weight in step2_weights:
				opw=opw*weight[index]
			instrument_and_weight[instrument].append(opw)

		for index,instrument in enumerate(step3):
			opw=1
			for weight in step3_weights:
				opw=opw*weight[index]
			instrument_and_weight[instrument].append(opw)

		#print i
		step_ordered=[]
		for instrument in instrument_and_weight.keys():
			#print instrument_and_weight[instrument]
			step_ordered.append([instrument, sum(instrument_and_weight[instrument])])
		step_ordered.sort(key=lambda x: x[1], reverse=True) #sort the list of instruments in descending order
		#print step_ordered

		######################## ACHTUNG! ##############################
		step_density=round((len(pattern1[i])*w1)+(len(pattern2[i])*w2)+(len(pattern3[i])*w3))
		# print i,step_density
		#attention: density is rounded using the "round" function. perhaps using "ceil" can be an alternative.
		###############################################################

		# pid1w1=[[x,w1] for x in pattern1[i]]
		# pid2w2=[[x,w2] for x in pattern2[i]]
		# pid3w3=[[x,w3] for x in pattern3[i]]

		
		# note_add= pid1w1+pid2w2+pid3w3 #make a list with all notes and their weights
		# note_set=set(pattern1[i]+pattern2[i]+pattern3[i]) #make a set with the sum of the notes in the step
		# #print "note set", note_set
		# #print "step weighted", note_add

		# #make a histogram with the notes present on the set
		# step_final=[]
		# for note in note_set:
		# 	common=[x for x in note_add if x[0] == note]
		# 	#print  common
		# 	common_added=sum([x[1] for x in common])
		# 	#print common_added
		# 	step_final.append([note,common_added])

		# step_final.sort(key=lambda x: x[1], reverse=True) #sort the histogram of instruments in descending order

		#print step_final

		#####################################################
		##################### OJO!!! ########################
		step_output=step_ordered[:int(step_density)]
		# when splitting a list in this way, we are assuming that the order of the onsets is already taken care of
		# but ITS NOT: for example if we have 5 different instruments at a given step with 1 repetition each
		# and the density is 3
		# How do we select which 3 of the list to output?
		# ideas: cerate an order 0 probability for each step, for all the patterns in the space.
		# in this way we can order the lists of same repetition instrument onsets with a criteria
		#####################################################

		step_output=[x[0] for x in step_output]
		if step_output==[]:
			step_output=[0]		
		#print i
		#print step_histogram, step_density
		#print step_output
		output_pattern.append(step_output)
	
	return output_pattern

def position2pattern(s, all_pattlists, pos, triangles, hashed_triangles, hash_density):
# input: search position, pattern positions, triangles, hashed_triangles, quadrant)
# output an interpolated pattern
	
	quadrant = pointer2hash(s, hash_density)

	sssss = search_in_space(s,triangles,pos,hashed_triangles,quadrant)

	i1,w1,i2,w2,i3,w3 = search_in_space(s,triangles,pos,hashed_triangles,quadrant)
	
	p1 = all_pattlists[int(i1)]
	p2 = all_pattlists[int(i2)]
	p3 = all_pattlists[int(i3)]
	
	return pattern_interpolation(p1,w1,p2,w2,p3,w3)