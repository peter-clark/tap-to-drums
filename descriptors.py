# this script contains descriptors meant to be extracted 
# form symbolic polyphonic drum patterns
# 
# the descriptors are coded as separated functions
# that can be run independently
# 
# most of them are reported in different papers 
# related to polyphonic drum analysis and generation
# * [1] "Similarity and Style in Electronic Dance Music Drum Rhythms"section 3.4
# * [2] "Strictly Rhythm: Exploring the Effects of Identical Regions and Meter Induction in Rhythmic Similarity Perception"
# * [3] "PAD and SAD: Two Awareness-Weighted Rhythmic Similarity Distances"
# * [4] "Drum rhythm spaces: From polyphonic similarity to generative maps"
# * [5] "Real-Time Drum Accompaniment Using Transformer Architecture"
# * [6] "Computational Creation and Morphing of Multilevel Rhythms by Control of Evenness"
# * [7] "The perceptual relevance of balance, evenness, and entropy in musical rhythms"
# * [8] "Syncopation, Body-Movement and Pleasure in Groove Music"

import numpy as np
from scipy import ndimage
from scipy import stats, integrate
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import random

###########################
# MIDI instrument mapping #
###########################
low_instruments=[35, 36, 41, 45, 47, 64]
mid_instruments=[37,38, 39, 40, 43, 48, 50, 58, 61, 62, 65, 77]
hi_instruments=[22,26, 42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 69, 70, 71, 72, 76]

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

def event_to_8number(midi_notes):
# input an event list and output a representation
# in 8 instrumental streams:
# kick, snare, rimshot, clap, closed hihat, open hihat, low tom, high tom
	output=[]
	# make sure the event has notes
	if len(midi_notes)>0:
		for x in midi_notes:
			#print("x", x)
			output.append(GM_dict[x][4])
	
	# otherwise it is a silence
		output=list(set(output))
		output.sort()
	else: 
		output=[0]

	return output

def event_to_3number(midi_notes):
# input an event list and output a representation
# in 3 instrumental streams:
# low, mid, high
	output=[]
	# make sure the event has notes
	if len(midi_notes)>0:
		for x in midi_notes:
			category=GM_dict[x][1]
			if category=="low":
				category_number=1
			elif category=="mid":
				category_number=2
			else:
				category_number=3
			output.append(category_number)
	
	# otherwise it is a silence
		output=list(set(output))
		output.sort()
	else: 
		output=[0]

	return output

##########################
# monophonic descriptors #
##########################

import math
import numpy as np

def density(patt):
# count the onsets in a pattern
	density= sum([x for x in patt if x==1])
	return density	

def syncopation16(patt):
# input a monophonic pattern as a list of 0s and 1s (1s indicating an onset)
# and obtain its syncopation value
	synclist=[0]*16
	salience_lhl=[5,1,2,1,3,1,2,1,4,1,2,1,3,1,2,1]
	for s,step in enumerate(patt):
		if patt[s]==1 and patt[(s+1)%len(patt)]==0: #look for an onset preceding a silence
			synclist[s]=salience_lhl[(s+1)%len(patt)]-salience_lhl[s] #compute syncopations

		output =sum(synclist) + 15
	
	return output

def syncopation16_awareness(patt):
# input a monophonic pattern as a list of 0s and 1s (1s indicating an onset)
# and obtain its awareness-weighted syncopation value
# awareness is reported in [2]
	synclist=[0]*16
	salience=[5,1,2,1,3,1,2,1,4,1,2,1,3,1,2,1]
	awareness=[5,1,4,2]
	for s,step in enumerate(patt):
		if patt[s]==1 and patt[(s+1)%16]==0: #look for an onset and a silence following
			synclist[s]=salience[(s+1)%16]-salience[s] #compute syncopations

	sync_and_awareness=[sum(synclist[0:4])*awareness[0],sum(synclist[4:8])*awareness[1],sum(synclist[8:12])*awareness[2], sum(synclist[12:16])*awareness[3]] # apply awareness
	output =sum(sync_and_awareness)
	
	return output

def evenness(patt):
# how well distributed are the D onsets of a pattern 
# if they are compared to a perfect D sided polygon
# input patterns are phase-corrected to start always at step 0
# i.e. if we have 4 onsets in a 16 step pattern, what is the distance of onsets
# o1, o2, o3, o4 to positions 0 4 8 and 12
# here we will use a simple algorithm that does not involve DFT computation
# evenness is well described in [6] but this implementation is much simpler
	if density(patt)!=0:
		dens = density(patt)
		iso_angle_16=2*math.pi/16
		first_onset_step=[i for i,x in enumerate(patt) if x==1][0]
		first_onset_angle=first_onset_step*iso_angle_16
		iso_angle=2*math.pi/dens
		iso_patt_radians=[x*iso_angle for x in range(dens)]
		patt_radians=[i*iso_angle_16 for i,x in enumerate(patt) if x==1]
		cosines=[abs(math.cos(x-patt_radians[i]+first_onset_angle)) for i,x in enumerate(iso_patt_radians)]
		evenness=sum(cosines)/dens
	else:
		evenness=0
	return evenness

def balance(patt):
# balance is described in [7] as:
# "a quantification of the proximity of that rhythm's
# “centre of mass” (the mean position of the points) 
# to the centre of the unit circle."
	center=np.array([0,0])
	iso_angle_16=2*math.pi/16
	X=[math.cos(i*iso_angle_16) for i,x in enumerate(patt) if x==1]
	Y=[math.sin(i*iso_angle_16) for i,x in enumerate(patt) if x==1]
	matrix=np.array([X,Y])
	matrixsum=matrix.sum(axis=1)
	magnitude=np.linalg.norm(matrixsum-center)/density(patt)
	balance=1-magnitude
	return balance

#########################
# polyphonic descriptors
#########################


def lowstream(pattlist):
# monophonic onset pattern of instruments in the low frequency range
	lowstream=[]
	for step in pattlist:
		step_result=0
		for instrument in step:
			if instrument in low_instruments:
				step_result=1
				break
		lowstream.append(step_result)
	return lowstream

def midstream(pattlist):
# monophonic onset pattern of instruments in the mid frequency range
	midstream=[]
	for step in pattlist:
		step_result=0
		for instrument in step:
			if instrument in mid_instruments:
				step_result=1
				break
		midstream.append(step_result)
	return midstream

def histream(pattlist):
# monophonic onset pattern of instruments in the hi frequency range
	histream=[]
	for step in pattlist:
		step_result=0
		for instrument in step:
			if instrument in hi_instruments:
				step_result=1
				break
		histream.append(step_result)
	return histream

def noi(pattlist):
# number of different instruments in a pattern
	noi=len(set([i for s in pattlist for i in s]))
	return noi

def loD(pattlist):
# density in the low frequency range
	loD=sum(lowstream(pattlist))
	return loD

def midD(pattlist):
# density in the mid frequency range
	midD=sum(midstream(pattlist))
	return midD

def hiD(pattlist):
# density in the hi frequency range
	if sum(histream(pattlist)) > 0:
		hiD=sum(histream(pattlist))
		return hiD
	else:
		return 0
def stepD(pattlist):
# percentage of steps that have onsets
	stepD=sum([1 for x in pattlist if x !=[]])/len(pattlist)
	return stepD

def lowness(pattlist):
# number of onsets in the low freq stream divided by the number of steps that have onsets
	if sum([1 for x in pattlist if x !=[]]) > 0:
		lowness=loD(pattlist)/sum([1 for x in pattlist if x !=[]])
	else:
		lowness=0
	return lowness

def midness(pattlist):
# number of onsets in the mid freq stream divided by the number of steps that have onsets
	if sum([1 for x in pattlist if x !=[]]) > 0:
		midness=midD(pattlist)/sum([1 for x in pattlist if x !=[]])
	else:
		midness=0
	return midness

def hiness(pattlist):
# number of onsets in the hi freq stream divided by the number of steps that have onsets
	if sum([1 for x in pattlist if x !=[]]) > 0:
		hiness=hiD(pattlist)/sum([1 for x in pattlist if x !=[]])
	else:
		hiness=0

	#print ("hiness", hiness)
	return hiness

def lowsync(pattlist):
# syncopation value of the low frequency stream
	lowsync=syncopation16(lowstream(pattlist))
	return lowsync

def midsync(pattlist):
# syncopation value of the mid frequency stream
	midsync=syncopation16(midstream(pattlist))
	return midsync

def hisync(pattlist):
# syncopation value of the high frequency stream
	hisync=syncopation16(histream(pattlist))
	return hisync

def losyness(pattlist):
# stream syncopation divided by the number of onsets of the stream
	if loD(pattlist)!=0:

		losyness=lowsync(pattlist)/loD(pattlist)
	else:
		losyness=0
	return losyness

def midsyness(pattlist):
# stream syncopation divided by the number of onsets of the stream
	if midD(pattlist)!=0:
		midsyness=midsync(pattlist)/midD(pattlist)
	else:
		midsyness=0
	return midsyness

def hisyness(pattlist):
# stream syncopation divided by the number of onsets of the stream
	if hiD(pattlist)!=0:
		hisyness=hisync(pattlist)/hiD(pattlist)
	else:
		hisyness=0
	return hisyness

def polysync(pattlist):
# polyphonic syncopation as described in [8]
# If N is a note that precedes a rest, R, 
# and R has a metric weight greater than or equal to N, 
# then the pair (N, R) is said to constitute a monophonic syncopation. 
# If N is a note on a certain instrument that precedes a note 
# on a different instrument (Ndi), and Ndi has a metric weight 
# greater than or equal to N, then the pair (N, Ndi) is said to
# constitute a polyphonic syncopation.

	salience_w= [0,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3] #metric profile as described by witek
	syncopation_list=[]
	# find pairs of N and Ndi notes events in the polyphonic pattlist
	for i in range(len(pattlist)):

		lowstream_=lowstream(pattlist)
		midstream_=midstream(pattlist)
		histream_=histream(pattlist)

		#describe the instruments present in current and nex steps
		event=[lowstream_[i],midstream_[i],histream_[i]] 
		event_next=[lowstream_[(i+1)%len(pattlist)],midstream_[(i+1)%len(pattlist)],histream_[(i+1)%len(pattlist)]]
		local_syncopation=0
		
		#syncopation: events are different, and next one has greater or equal metric weight
		if event!= event_next and salience_w[(i+1)%len(pattlist)]>=salience_w[i]: # only process if there is a syncopation
			# now analyze what type of syncopation is found to assign instrumental weight
			# instrumental weight depends on the relationship between the instruments in the pair:
			
			##### three-stream syncopations:
			# low (event[0]) against mid and hi (event_next[1] and event_next[2] respectively)
			if event[0]==1 and event_next[1]==1 and event_next[2]==1:
				instrumental_weight = 2
				local_syncopation = abs(salience_w[i]-salience_w[(i+1)%len(pattlist)]) + instrumental_weight
				
			# mid syncopated against low and high
			# mid (event[1]) against low and hi (evet_next[0] and event_next[2] respectively)
			if event[1]==1 and event_next[0]==1 and event_next[2]==1:
				instrumental_weight=1
				local_syncopation = abs(salience_w[i]-salience_w[(i+1)%len(pattlist)]) + instrumental_weight
				
			##### two stream syncopations:
			# low or mid vs high
			if (event[0]==1 or event[1]==1) and event_next==[0,0,1]:
				instrumental_weight=5
				local_syncopation = abs(salience_w[i]-salience_w[(i+1)%len(pattlist)]) + instrumental_weight
				
			# low vs mid (ATTENTION: not on Witek's paper)
			if event==[1,0,0] and event_next==[0,1,0]:
				instrumental_weight = 2
				local_syncopation = abs(salience_w[i]-salience_w[(i+1)%len(pattlist)]) + instrumental_weight

			# mid vs low (ATTENTION: not on Witek's paper)
			if event==[0,1,0] and event_next==[1,0,0]:
				instrumental_weight = 2
				local_syncopation = abs(salience_w[i]-salience_w[(i+1)%len(pattlist)]) + instrumental_weight

			syncopation_list.append(local_syncopation)
	#print("list", syncopation_list)
	polysync=sum(syncopation_list)
	return polysync

def polyevenness(pattlist):
# compute the polyphonic evenness of a pattlist
# adapted from [7]
	lowstream_=lowstream(pattlist)
	midstream_=midstream(pattlist)
	histream_=histream(pattlist)

	low_evenness=evenness(lowstream_)
	mid_evenness=evenness(midstream_)
	hi_evenness=evenness(histream_)
	
	polyevenness=low_evenness*3+mid_evenness*2+hi_evenness

	return polyevenness

def polybalance(pattlist):
#compute the polyphonic balance of a pattlist
# adapted from [7]
	lowstream_=lowstream(pattlist)
	midstream_=midstream(pattlist)
	histream_=histream(pattlist)
	alldensity=density(lowstream_)*3+density(midstream_)*2+density(histream_)
	if alldensity != 0:
		center=np.array([0,0])
		iso_angle_16=2*math.pi/16
		
		Xlow=[3*math.cos(i*iso_angle_16) for i,x in enumerate(lowstream_) if x==1]
		Ylow=[3*math.sin(i*iso_angle_16) for i,x in enumerate(lowstream_) if x==1]
		matrixlow=np.array([Xlow,Ylow])
		matrixlowsum=matrixlow.sum(axis=1)

		Xmid=[2*math.cos(i*iso_angle_16) for i,x in enumerate(midstream_) if x==1]
		Ymid=[2*math.sin(i*iso_angle_16) for i,x in enumerate(midstream_) if x==1]
		matrixmid=np.array([Xmid,Ymid])
		matrixmidsum=matrixmid.sum(axis=1)

		Xhi=[2*math.cos(i*iso_angle_16) for i,x in enumerate(histream_) if x==1]
		Yhi=[2*math.sin(i*iso_angle_16) for i,x in enumerate(histream_) if x==1]
		matrixhi=np.array([Xhi,Yhi])
		matrixhisum=matrixhi.sum(axis=1)

		matrixsum=matrixlowsum+matrixmidsum+matrixhisum
		magnitude=np.linalg.norm(matrixsum-center)/alldensity
	else:
		return 1
	balance=1-magnitude
	return balance

def polyD(pattlist):
# compute the total number of onsets
	return loD(pattlist)+midD(pattlist)+hiD(pattlist)

def pattlist2descriptors(pattlist):
# compute all descriptors from a polyphonic drum pattern.
	all_descriptors=[
	noi(pattlist),
	loD(pattlist),
	midD(pattlist),
	hiD(pattlist),
	stepD(pattlist),
	lowness(pattlist),
	midness(pattlist),
	hiness(pattlist),
	lowsync(pattlist),
	midsync(pattlist),
	hisync(pattlist),
	losyness(pattlist),
	midsyness(pattlist),
	hisyness(pattlist),
	polysync(pattlist),
	polyevenness(pattlist),
	polybalance(pattlist),
	polyD(pattlist)]
	
	#print ("all_descriptors", all_descriptors)
	
	return all_descriptors

# this is a useful list for making plots and extrcating infromation
descriptor_name=[
	"noi",
	"loD",
	"midD",
	"hiD",
	"stepD",
	"lowness",
	"midness",
	"hiness",
	"lowsync",
	"midsync",
	"hisync",
	"losyness",
	"midsyness",
	"hisyness",
	"polysync",
	"polyevenness",
	"polybalance",
	"polyD"]

def lopl2descriptors(list_of_pattlists):
# compute all descriptors from a list of pattlists
# return a numpy array with columns as descriptors
# and lines as patterns.

	array=np.empty((len(list_of_pattlists), len(descriptor_name))) # create an empty ndarray
	for i,pattlist in enumerate(list_of_pattlists):
		desc_array=np.array([pattlist2descriptors(pattlist)]) #notice the extra [] adds dimension
		#print ("descriptor array", desc_array)
		array[i]=desc_array # fill up the array
	return array

def overlapping_area(A, B):
# input wo data series, create pdfs
# return the overlapping area

	# obtain density functions from data series
    pdf_A = stats.gaussian_kde(A) 
    pdf_B = stats.gaussian_kde(B)

    lower_limit = np.min((np.min(A), np.min(B))) # min value of both series
    upper_limit = np.max((np.max(A), np.max(B))) # max value from both series

    area = integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)),lower_limit ,upper_limit )[0] #make integration

    return area 

def kl_dist(A, B, num_sample=1000):
# input two data series, create pdfs
# A is the reference (training set)
# B is the pairwise comparison between A (training set) and B (model output)
# return entropy (A/B)

    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)

    entropy=stats.entropy(pdf_A(sample_A), pdf_B(sample_B))

    return entropy

def smooth_curve(curve):
# input a distribution with sampling errors (y=0)
# and output a smoothed version without errors
    pairs=[(i,x) for i,x in enumerate(curve) if x!= 0 and curve[(i+1)%len(curve)] ==0]
    for x in range(len(pairs)-1):
        xa=pairs[x][0]
        xb=pairs[x+1][0]
        ya=pairs[x][1]
        yb=pairs[x+1][1]
        for f in range(xb-xa-1):
            curve[f+xa+1]=ya+(((yb-ya)/(xb-xa))*f)
    return curve

def smooth_col(col, bins):
# input a numpy array of descriptor values 
# output: 
# a resampled numpy array
# the original pairwise histogram
# a smoothed KDE

	#print("col", col)
	# uncomment plots to see example
	x_range=np.arange(0,1, 1/bins)
    # basic descriptor values
	basic_hist=ndimage.histogram(col, 0, 1, bins) 
    # normalize area from 0 to bin so probabilities from 0 to 1 fit
    #basic_hist=basic_hist/sum(basic_hist)
    # plt.plot(x_range,basic_hist, color="red", label="original histogram")
    #print("basic_hist", basic_hist)
    #####################################################
    #         create pairwise dissim values for col     #
    #      comment this section for fast computation    #
    #         uncomment to see original histogram       #
    #####################################################    

    # # descriptor values dissim
    # # dissim matrix (0 to 1)
    # pairwise_col=pairwise_distances(col.reshape(-1,1), Y=None, metric='euclidean')
    # #flatten matrix
    # flat_pairwise_col=pairwise_col.flatten()
    # # histogram (x range 0 to 1)
    # col_hist=ndimage.histogram(flat_pairwise_col, 0, 1, bins) 
    # # normalize area from 0 to bin
    # col_hist=col_hist/max(col_hist)
    # plt.plot(x_range,col_hist, color="blue", label="original pairwise histogram")
    
    ##################################################### 

    #####################################################
    #         create resampled values for col           #
    #####################################################

    ### smooth the original histogram (aka basic hist)
	smooth_hist=np.array(smooth_curve(basic_hist))
    # normalize smoothed hist
    #smooth_hist=np.array(smooth_hist)/max(smooth_hist)

    # plt.plot(x_range,smooth_hist, color="green", label="smoothed original histogram")

    # normalize smooth_hist to be area = 1
	smooth_hist=smooth_hist/sum(smooth_hist)
    #print("smooth_hist", smooth_hist)
    # resample descriptor values based on the smoothed original histogram
    # save as "new_col"
	new_col=[]
    
	for x in range(len(col)): # make as much iterations as descriptors
		ran=random.random() #gen a random number
        # search for the index where the area of this random number lays
        # print ("ran", ran)
		index=0
		area=0
		while area <= ran: # while the random number is larger than the accum area
			index += 1 # increase index 
			area += smooth_hist[index%len(smooth_hist)]
		new_col.append(index/bins) #append denorm value
        #print(ran,index, area, smooth_hist[index%len(smooth_hist)])
	new_col=np.array(new_col)
    #####################################################

    #####################################################
    # make a new col histogram to compare
    # this section can be commented for faster computation
    #####################################################

    # # dissim histogram
    # new_basic_hist=ndimage.histogram(new_col, 0, 1, bins) 
    # # normalize area from 0 to bin
    # new_basic_hist=new_basic_hist/max(new_basic_hist)

    # # plt.plot(x_range,new_basic_hist, color="black", label="new original histogram")
    #####################################################

    #####################################################
    # create a new dissim histogram
    #####################################################
    # dissim matrix (0 to 1)
	#print("new_col", new_col)
	pairwise_col=pairwise_distances(new_col.reshape(-1,1), Y=None, metric='euclidean')
	#print("pairwise_col", pairwise_col)
    #flatten matrix
	flat_pairwise_col=pairwise_col.flatten()
    # histogram (x range 0 to 1)
	new_col_hist=ndimage.histogram(flat_pairwise_col, 0, 1, bins) 
    # normalize area from 0 to 1
	new_col_hist=new_col_hist/sum(new_col_hist)

    # plt.plot(x_range,new_col_hist, color="purple", label="smoothed pairwise histogram")
    #####################################################

    #####################################################
    # finally compute the kde
    #####################################################
	#print("fpwc",flat_pairwise_col)
	if sum(flat_pairwise_col)==0:
		flat_pairwise_col[0] = 0.00001 # in case a column is all 0
	col_kde=stats.gaussian_kde(np.transpose(flat_pairwise_col))
	# print (col_kde)
    # #col_kde=stats.gaussian_kde(new_col)
    # normalize area from 0 to 1
	kde_curve=col_kde(x_range)/sum(col_kde(x_range))
    #plt.plot(x_range, kde_curve, color="cyan", label="smoothed pairwise kde")
    #plt.legend(loc = 'upper right')
    #plt.show()

    # the KDE and the histogram will be normalized to area
	return new_col, new_col_hist, kde_curve

def dmatrix_distribution(style,desc_array, folder, bins, plot):
# input a descriptor matrix as a numpy ndarray
# and generate:
# 1 the histograms of the pairwise comparisons of each descriptor
# 2 the kdes of the pairwise comparison of each descriptor
# 3 the histograms of the pairwise comparison among the samples (all descriptors)
# 4 the kde of the pairwise comparison among the samples (all descriptors)
# 5 the normalized descriptor matrix (useful for comparing observed and generated)
# The number of bins can be specified for different detail levels.
# This is useful for visualizing a collection in terms
# of the polyphonic descriptors presented above.
	
	# normalize the distribution
	# desc_array = desc_array-desc_array.min(axis=0) #subtract lowest value
	model_desc_norm = desc_array / desc_array.max(axis=0) #divide by max
	model_desc_norm[np.isnan(model_desc_norm)] = 0 # avoid nans, convert to 0 (i.e. a row only has 0/0)

	# pairwise comparison among patterns (creates a dissimilarity matrix)
	# print("model_desc_norm", model_desc_norm)
	pairwise_mtx=pairwise_distances(model_desc_norm, Y=None, metric='euclidean')

	# flatten the dissim matrix before computing the histogram
	model_pw=pairwise_mtx.flatten()

	#print ("model_pw", model_pw)

	# apply scipy.ndimage.histogram to create a histogram
	model_histogram=ndimage.histogram(model_pw, 0, 10, bins) #min=0, max=1, bins=bins

	# normalize the histogram so area = 1
	model_histogram=model_histogram/np.sum(model_histogram)

	# apply kernel density estimation to obtain a kde
	model_kde=stats.gaussian_kde(np.transpose(model_pw)) # transpose column so it is a 1d np.array

	# find the number of columns in the matrix
	col_num = np.shape(model_desc_norm)[1]

	#create an ndarray to store the histograms
	desc_pw=np.empty((col_num, bins)) #note that each row is now a descriptor

	# create a list to store the KDEs of every column
	desc_pw_kde=[] #each element is a descriptor

	# extract each column of the model_desc_norm
	for col_index in range(col_num):
		col=model_desc_norm[:,[col_index]] #extract the normalized column
		#print("descriptor_name", descriptor_name[col_index])
		
		new_col, col_pws_histogram, col_kde  = smooth_col(col, bins)

		# new_col=col
		# col_pws_histogram=
		# col_kde=


		# create a dissimiality matrix of the column by pairwise comparison
		#pairwise_col=pairwise_distances(col.reshape(-1,1), Y=None, metric='euclidean')
		# flatten the dissim matrix
		#flat_pairwise_col=pairwise_col.flatten()
		# apply scipy.ndimage.histogram to create a histogram
		#col_histogram=ndimage.histogram(flat_pairwise_col, 0, bins, bins) #min=0, max=bins, bins=bins
		# normalize the histogram so area = 1
		#col_histogram=col_histogram/np.sum(col_histogram)
		# apply kernel density estimation to obtain a kde
		#col_kde=stats.gaussian_kde(np.transpose(flat_pairwise_col)) # transpose column so it is a 1d np.array

		desc_pw[col_index]=col_pws_histogram
		desc_pw_kde.append(col_kde)
		if plot==True:  #if histograms are to be plotted
			fig = plt.figure(figsize = (6, 3))
			#x_range=[str(x/bins) for x in range(bins)]
			x_range=np.arange(0,1,1/bins)
			#col_width=bins/sum([1 for x in col_histogram if x!=0]) # use this variable to amplify the kde
			#plt.hist(flat_pairwise_col, bins, density=True)
			plt.plot(x_range, col_pws_histogram, label="pairwise distance histogram")
			plt.plot(x_range, col_kde, label="pairwise distance kde")
			plt.xlabel("value")
			plt.ylabel("distribution")
			plt.title(style+" Style. Distribution of Descriptor "+descriptor_name[col_index])
			plt.xlim(0,1)
			plt.savefig("analysis/"+folder+"/"+style+"/"+descriptor_name[col_index])
			plt.legend(loc = 'upper right')
			plt.show()
			# save the figure as the style name in the folder defined by the user
	
	# model_desc_norm : column-normalized model descriptors
	# model_pw : intra model pairwise comparison
	# model_kde : resulting intra kdes 
	# desc_pw : pairwise comparison of each descriptor column
	# desc_pw_kde : resulting kdes of each descriptor (new)

	# return the histograms, the kdes and the normalized descriptor matrix
	return desc_pw, desc_pw_kde, model_histogram,  model_kde, model_desc_norm, model_pw

def inter_comparison(trn_model_desc_norm, trn_model_pw, gen_model_desc_norm, model_id, gen_model_pw, bins):
# compare the original model with the generated model 
# output the inter pairwise combination and the inter KDE
	
	# data_dicitonary
	dd={}

	# create the inter distribution 
	# compare each gen element vs each train element
	inter_pw=np.array([np.linalg.norm(a-b)for a in trn_model_desc_norm for b in gen_model_desc_norm])
	flat_inter_pw=inter_pw.flatten()
	inter_kde = stats.gaussian_kde(np.transpose(flat_inter_pw))
	inter_hist=ndimage.histogram(flat_inter_pw, 0, 10, bins) #min=0, max=10, bins=bins
	inter_hist=inter_hist/np.sum(inter_hist) # normalize so area = 1 (as pdf)
	
	# compare original set vs generated set
	x=np.arange(0,1, 1/bins)
	model_mean=np.mean(trn_model_pw)
	model_std=np.std(trn_model_pw)
	gen_mean_o=np.mean(gen_model_pw)
	gen_std_o=np.std(gen_model_pw)
	inter_mean_o=np.mean(flat_inter_pw)
	inter_std_o=np.std(flat_inter_pw)
	#model_OA_o = np.sum(np.minimum(inter_kde(x)/np.sum(inter_kde(x)),trn_model_kde(x)/np.sum(trn_model_kde(x))))
	#model_kld_o = np.sum([ x for x in rel_entr(inter_kde(x)/np.sum(inter_kde(x)), trn_model_kde(x)/np.sum(trn_model_kde(x))) if x!="inf"])

	model_OA_o=overlapping_area(trn_model_pw, flat_inter_pw)
	model_kld_o=kl_dist(trn_model_pw, flat_inter_pw)

	# save in the data dictionary
	dd["model mean"]= model_mean # save to output
	dd["model std"]=model_std # save to output
	dd["gen mean o"+str(model_id)]=gen_mean_o # save to output
	dd["gen std o"+str(model_id)]=gen_std_o # save to output
	dd["inter mean o"]=inter_mean_o # save to output
	dd["inter std o"+str(model_id)]=inter_std_o # save to output
	dd["model OA o"+str(model_id)]=model_OA_o
	dd["model kld o"+str(model_id)]=model_kld_o

	return inter_hist, model_OA_o, model_kld_o, dd



##################################
# uncomment below for an example #
##################################
# pattlist=[[36,42],[],[42], [37], [36,42],[],[42,37], []]
# print (pattlist2descriptors(pattlist))
