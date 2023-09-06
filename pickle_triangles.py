import pickle
import functions as fun
import descriptors as desc
import os

dir = os.getcwd()
all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments") # Parse all MIDI patterns in folders
print("Extracted Patterns.\n")

d = desc.lopl2descriptors(all_pattlists) 
print("Got polyphonic descriptors.\n")
mds_pos= fun.d_mtx_2_mds(d)
print("Got MDS embedding coordinates.\n")

## LOAD TRIANGLES, HASH_TRIANGLES, HASH_DENSITY
triangles = fun.create_delaunay_triangles(mds_pos) #create triangles
print("Calculated Triangles.\n")
hash_density = 2 # number of col, row in space
hashed_triangles = fun.hash_triangles(mds_pos, triangles, hash_density) # 2 means divide the space in 2 cols and 2 rows
print("Triangle Hashing.\n")

## Pickle everything for later use
save_dir = dir + "/data/"
# Patterns & Names
file = open(save_dir+"patterns.pkl", 'wb')
pickle.dump(all_pattlists, file, -1)
file.close()
file = open(save_dir+"pattern_names.pkl", 'wb')
pickle.dump(all_names, file, -1)
file.close()

# Descriptors
file = open(save_dir+"descriptors.pkl", 'wb')
pickle.dump(d, file, -1)
file.close()

# Coordinates
file = open(save_dir+"mds_pos.pkl", 'wb')
pickle.dump(mds_pos, file, -1)
file.close()

# Triangles
file = open(save_dir+"triangles.pkl", 'wb')
pickle.dump(triangles, file, -1)
file.close()

# Hashed Triangles
file = open(save_dir+"hashed_triangles.pkl", 'wb')
pickle.dump(hashed_triangles, file, -1)
file.close()