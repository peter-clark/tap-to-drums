import os
import numpy as np
import descriptors as desc
import sklearn
import re
# Regression Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

import pickle


# This function regex parses a line in a txt file to
#   extract all the floats and integers in an input.
#   The output is an array of floats. 
def parse(line):
    line = str(line)
    regex = r"[-+]?\d*\.\d+|\d" # searches for all floats or integers
    list = re.findall(regex, line)
    output = [float(x) for x in list]
    return output

#--------------------------------------------------------------------------------------#
flat = []
mds_pos = []

dir_list = os.listdir(os.getcwd())
pickle_dir = os.getcwd()+"/data/"
""" for item in dir_list:
    print(item)
filename = input("Load Patterns File: ")
with open(filename) as file:
    file_contents=[]
    for line in file:
        l = parse(line)
        file_contents.append(l)
    flat=file_contents
filename = input("Load Position:(X,Y) File: ")
with open(filename) as file:
    file_contents=[]
    for line in file:
        l = parse(line)
        file_contents.append(l)
    mds_pos=file_contents """

mds_pos_file = open(pickle_dir+"mds_pos.pkl", 'rb')
mds_pos = pickle.load(mds_pos_file) 
mds_pos_file.close()
flat_file =open(os.getcwd()+"/flat/flatbyalg.pkl",'rb')
flat = pickle.load(flat_file)
flat_file.close()
print(len(flat))
print(len(flat[0]))
print(len(flat[0][0]))
print(len(mds_pos))
print(len(mds_pos[0]))
temp = flat
flat = temp[5]
# Regressions
linear_model = LinearRegression()
linear_model.fit(flat,mds_pos)

kNN_model = KNeighborsRegressor()
kNN_model.fit(flat,mds_pos)

DT_model = DecisionTreeRegressor()
DT_model.fit(flat,mds_pos)

svr_model = LinearSVR()
svr_wrap_model = MultiOutputRegressor(svr_model)
svr_wrap_model.fit(flat,mds_pos)

#   Define procedure
cross_validation = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1) # 10-fold
#   Evaluate
np.seterr("ignore")
n_scores_linear = np.abs(cross_val_score(linear_model, flat, mds_pos, scoring='neg_mean_absolute_error', cv=cross_validation,n_jobs=-1))
n_scores_kNN = np.abs(cross_val_score(kNN_model, flat, mds_pos, scoring='neg_mean_absolute_error', cv=cross_validation,n_jobs=-1))
n_scores_DT = np.abs(cross_val_score(DT_model, flat, mds_pos, scoring='neg_mean_absolute_error', cv=cross_validation,n_jobs=-1))
n_scores_svr = np.abs(cross_val_score(svr_wrap_model, flat, mds_pos, scoring='neg_mean_absolute_error', cv=cross_validation,n_jobs=-1))

print("Mean Abs Error[Linear]: %.3f (%.3f)" % (np.mean(n_scores_linear), np.std(n_scores_linear)))
print("Mean Abs Error[kNN]: %.3f (%.3f)" % (np.mean(n_scores_kNN), np.std(n_scores_kNN)))
print("Mean Abs Error[DT]: %.3f (%.3f)" % (np.mean(n_scores_DT), np.std(n_scores_DT)))
print("Mean Abs Error[SVR]: %.3f (%.3f)" % (np.mean(n_scores_svr), np.std(n_scores_svr)))
#

