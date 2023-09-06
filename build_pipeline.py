import os
import sys
import numpy as np
import functions as fun
import descriptors as desc
import flatten as flatten
import neuralnetwork as NN
import time
import csv
import pickle
s = time.perf_counter()

## Initalize variables
dir = os.getcwd()
_savepatterns = False
_saveflattened = False
_saveembeddings = False
_savemodels = True
_savepredictions = True
#_savepredictions = _savemodels

## Extract all patterns and names from MIDI files in (+sub)folder
all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments")
patterns_dir = dir + "/patterns/"
print("\nExtracted Patterns\n")

if _savepatterns:
    for index in range(len(all_pattlists)):
        filename = patterns_dir+str(all_names[index])+".txt"
        with open(filename, "w") as f:
            f.write(str(all_pattlists[index]))

## Get polyphonic descriptors for patterns
d = desc.lopl2descriptors(all_pattlists)

## Slice for 5 significant descriptors
_d = np.asarray([np.asarray([de[2],de[3],de[7],de[8],de[13]]) for de in d])

print("Calculated Polyphonic Descriptors \n")


## Get coordinates from embedding done on poly-descriptors
#       [MDS, PCA, TSNE, UMAP]
embeddings_dir = dir + "/embeddings/"
embeddings = []
embeddings_names = ["MDS","PCA","TSNE","UMAP"]

mds_pos = fun.d_mtx_2_mds(d)
embeddings.append(mds_pos)
print("Got embedding coordinates\n")

#   Save if desired
if _saveembeddings:
    for i in range(len(embeddings)):
        filename = embeddings_dir + embeddings_names[i] + ".txt"
        filename_csv = embeddings_dir + embeddings_names[i] + ".csv"
        with open(filename,"w") as f:
            g=open(filename_csv,'w')
            writer = csv.writer(g)
            for pos in range(len(embeddings[i])):
                writer.writerow(embeddings[i][pos])
                f.write(str(embeddings[i][pos])+"\n") if pos != len(embeddings[i]-1) else f.write(str(embeddings[i][pos]))
            g.close()

## Apply flattening algorithms to all patterns
#       [continuous1, continuous2, discrete1, discrete2]
flat_names = ["continuous1", "discrete1", "continuous2", "discrete2","semicontinuous1", "semicontinuous2"]
flattened_dir = dir + "/flattened/"
all_flat = [[] for x in range(len(all_pattlists))]
flat_by_alg = [[[] for x in range(len(all_pattlists))] for y in range(6)]
for pattern in range(len(all_pattlists)):
    #print(all_names[pattern])
    flat = flatten.flat_from_patt(all_pattlists[pattern])
    #print(len(flat))
    #print(len(flat[1]))
    sc1 = np.where(flat[1]==1, flat[0],flat[1])
    sc2 = np.where(flat[3]==1, flat[2],flat[3])
    #flat.append(sc1)
    #flat.append(sc2)
    all_flat[pattern] = flat
    for i in range(len(flat_by_alg)):
        flat_by_alg[i][pattern] = flat[i]

#   Save if desired
    if _saveflattened:
        filename = flattened_dir+str(all_names[pattern])+".txt"
        with open(filename, "w") as f:
            for i in range(len(flat)):
                f.write(str(all_flat[pattern][i])+"\n") if i!=len(flat)-1 else f.write(str(all_flat[pattern][i]))
        for i in range(len(flat_by_alg)):
            with open(dir+"/flat/"+flat_names[i]+".txt",'w') as g:
                for pattern in range(len(all_pattlists)):
                    g.write(str(flat_by_alg[i][pattern])+"\n") if i!=len(all_pattlists)-1 else g.write(str(flat_by_alg[i][pattern]))
file = open(os.getcwd()+"/flat/flatbyalg.pkl", 'wb')
pickle.dump(flat_by_alg, file, -1)
file.close()
print("Patterns have been flattened\n")


## Send flattened patterns + embedding coordinates to model to train
#       (4 x 4) -> embeddings x patterns 
#       - save models once trained
model_dir = dir + "/models/"
for embed in embeddings:
    for alg in range(len(flat_by_alg)):
        predicted_coords = []
        # Build model
        model_dir += (flat_names[alg])
        print(flat_names[alg]+"--------------")
        predicted_coords = NN.NN_pipeline(flat_by_alg[alg], embed, _savemodels, model_dir)
        #predicted_coords = NN.NN_pipeline(flat_by_alg[alg], embed, _savemodels, model_dir, True)
        model_dir=dir + "/models/"
        if _savepredictions:
            with open(dir+"/predictions/"+flat_names[alg]+".csv",'w') as f:
                writer = csv.writer(f)                
                for i in range(len(predicted_coords)):
                    writer.writerow(predicted_coords[i])
                    #f.write(str(predicted_coords[i])+"\n") if i!=len(predicted_coords)-1 else f.write(str(predicted_coords[i]))

print(f"Runtime: {time.perf_counter()-s:.2f} seconds")
