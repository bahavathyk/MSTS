# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:22:40 2020

@author: kbaha
"""

# Import Required Packages
import numpy as np
import matplotlib
import pandas as pd
import math
from tslearn.metrics import dtw, cdist_dtw
import itertools 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
import time

from sklearn.model_selection import train_test_split
import sklearn
# Import the functions required 
from MSFunctions.MSFunctions import Calc_MIScore, find_dtw_dist, knn_clfcrossvalxv, knn_clfcrossvalxv_precomp, Create_input, Cal_numer, FindMerit_Score, Create_input_selected, Findaccuracy, FindCombsaccuracy
from MSFunctions.Load_Data import FindUEAXy, list_files

#%%
#For UEA Datasets

Dataset_Name = 'RacketSports'

# Set the name of the Dataset
files = list_files(r"TSDatasets\\"+Dataset_Name+"\MTSDATA\Train")
files_test = list_files(r"TSDatasets\\"+Dataset_Name+"\MTSDATA\Test")

X_full, y_true, target = FindUEAXy(files)

# Find size of X
n,m,l = X_full.shape

writefile = open("Results/"+Dataset_Name+"_DTW.txt", "w+")

# Calculate the distances and save in advance
dist = []
mi_kr = []
for i in range(0,n):
	print(i)
	dist.append(cdist_dtw(X_full[i,:,:], X_full[i,:,:]))
	print(dist)

np.set_printoptions(threshold=np.inf)
print(dist, file = writefile)

writefile.close()	

#%% 
start_MS = time.process_time()  

# The classes for the predictions
data_classes = target

# Make single feature predictions
y_pred = pd.DataFrame()
acc = []
time_acc_MS_1 = 0
for i in range(0,n):
	
	knn_clf = KNeighborsClassifier(n_neighbors=1, metric = 'precomputed')
	y_prediction = cross_val_predict(knn_clf, dist[i],y_true, cv=5)
	y_pred.insert(i,i,y_prediction)
	 
	start_MS_acc_1 = time.process_time()
	y_acc = cross_val_score(knn_clf, dist[i],y_true, cv=5, scoring = 'accuracy')
	acc.append(y_acc.mean())
	timetaken_MS_acc_1 = time.process_time() - start_MS_acc_1
	time_acc_MS_1 = time_acc_MS_1+timetaken_MS_acc_1
 

# MI for each variable compared with true value
y_predicted_label = pd.DataFrame()
MI_y = pd.DataFrame()
for i in range(0,n):
    y_predicted_label.insert(i,i,y_pred[i].apply(data_classes.index))
    MI_y.loc[i,0] = adjusted_mutual_info_score(y_pred[i], y_true)
	 
MI_knn = sklearn.feature_selection.mutual_info_classif(X_full[:,:,0].T, y_true, n_neighbors = 3)
#kraskov_mi(X_full[0,:,:]) 

# No of variables to initially use
var_no = 2

# initialise variables
sel_comb = []
merit_score_change = 1
merit_score_prev = 0
accuracy_sel_MS = pd.DataFrame()
merit_score_sel = pd.DataFrame()
var_no_df_MS = pd.DataFrame()
time_acc_MS = 0
enum = 0
f_scatter = plt.figure(dpi = 300)

writefile = open("Results/"+Dataset_Name+".txt", "w+")


for  i in range(0,n-1):
	merit_score, combs = FindMerit_Score(MI_y,var_no, n, sel_comb, y_predicted_label, X_full)
	merit_score_change = max(merit_score) - merit_score_prev
	
	start_MS_acc = time.process_time()  
	accuracy = FindCombsaccuracy(var_no, combs, dist, y_true)
	timetaken_MS_acc = time.process_time() - start_MS_acc
	time_acc_MS = time_acc_MS+timetaken_MS_acc
	
	if(merit_score_change <= 0):
		break
	else:
		sel_comb = combs[np.argmax(merit_score)]
		merit_score_prev = max(merit_score)
		subset_size_MS = var_no
		var_no_df_MS.insert(enum, enum,[var_no])
		var_no = var_no + 1
		
		plt.scatter(accuracy, merit_score)
		accuracy_sel_MS.insert(enum, enum,[accuracy[np.argmax(merit_score)]])
		merit_score_sel.insert(enum, enum,[ merit_score_prev])
		enum = enum+1
		MS_bestacc = accuracy[np.argmax(merit_score)]
		print("MeritScore Results", file = writefile)
		print("Best accuracy", accuracy[np.argmax(merit_score)], file = writefile)
		print("Best merit_score", merit_score_prev, file = writefile)

writefile.close()	

timetaken_MS = time.process_time() - start_MS
totaltime_MS = timetaken_MS - time_acc_MS - time_acc_MS_1

#plt.figure(dpi = 300)
X_lab = plt.scatter(accuracy_sel_MS, merit_score_sel, marker = 'x', c = 'k', label = 'Selected Subset')
plt.xlabel("accuracy", fontsize = 12)
plt.ylabel("MSTS", fontsize = 12)
plt.title(Dataset_Name, fontsize = 14)
plt.grid('major', c = 'grey', linewidth = 0.1)
plt.legend([X_lab], ["Selected Subset of size n"])
	
plt.savefig("Results/"+Dataset_Name+".png") 
	
# initialise variables
start_GS = time.process_time() 
#sel_comb_acc = [np.argmax(acc)+1]
sel_comb_acc = []
acc_score_change = 1
acc_score_prev = 0
accuracy_sel = pd.DataFrame()
var_no_df = pd.DataFrame()
enum = 0
plt.figure(dpi = 300)

writefile = open("Results/"+Dataset_Name+".txt", "a")
var_no = 2
for  i in range(0,n-1):
	accuracy, combs = Findaccuracy(var_no, n, sel_comb_acc, X_full, dist, y_true)
	acc_score_change = max(accuracy) - acc_score_prev
	
	if(acc_score_change <= 0):
		break
	else:
		sel_comb_acc = combs[np.argmax(accuracy)]
		acc_score_prev = max(accuracy)
		subset_size_acc = var_no
		var_no_df.insert(enum, enum,[var_no])
		var_no = var_no + 1
		
		#plt.scatter(var_no, accuracy)
		accuracy_sel.insert(enum, enum,[accuracy[np.argmax(accuracy)]])
		enum = enum+1
		GS_bestacc = accuracy[np.argmax(accuracy)]
		print("\nGreedy Search Results", file = writefile)
		print("Best accuracy", accuracy[np.argmax(accuracy)], file = writefile)
		print("Best accuracy found", acc_score_prev, file = writefile)

totaltime_GS = time.process_time() - start_GS + time_acc_MS_1
writefile.close()

plt.figure(dpi = 300)
method = ["MeritScore", "GreedySearch"]
plt.bar(method, [MS_bestacc, GS_bestacc])
plt.title(Dataset_Name)
plt.ylabel("Accuracy")

plt.figure(dpi = 300)
plt.scatter(var_no_df, accuracy_sel)
plt.scatter(var_no_df_MS, accuracy_sel_MS)
plt.xlabel("number of variables in subset")
plt.ylabel("Accuracy")
plt.title(Dataset_Name)
plt.legend(["Greedy Search", "Merit Search"], loc = 'upper left')

writefile = open("Results/"+Dataset_Name+".txt", "a")
print("Time taken to identify best subset using Merit score: ", totaltime_MS, file = writefile)
print("Time taken to identify best subset using Greedy search: ", totaltime_GS, file = writefile)

#%% Evaluate on Test data

#For UEA Datasets

# Train data using selected combination of features

X_train, y_train, y_target = FindUEAXy(files)
a,b,c = X_train.shape
y_train = y_train[0:b].astype(str)
y_train.shape 
X_train_input = Create_input_selected(len(sel_comb), X_train, sel_comb)

knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
knn_clf.fit(X_train_input,y_train)

# Test data

X_test, y_test, y_target = FindUEAXy(files_test)
a,b,c = X_test.shape
y_test = y_test[0:b].astype(str)
y_test.shape 

X_test_input = Create_input_selected(len(sel_comb), X_test, sel_comb)
y_pred = knn_clf.predict(X_test_input)

# evaluate predictions
acc = accuracy_score(y_test, y_pred)
print("best accuracy on holdout, merit selected subset is", acc, file = writefile)

# For greedy search selected 
X_train, y_train, y_target = FindUEAXy(files)
a,b,c = X_train.shape
y_train = y_train[0:b].astype(str)
y_train.shape 
X_train_input = Create_input_selected(len(sel_comb_acc), X_train, sel_comb_acc)

knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
knn_clf.fit(X_train_input,y_train)

# Test data

X_test, y_test, y_target = FindUEAXy(files_test)
a,b,c = X_test.shape
y_test = y_test[0:b].astype(str)
y_test.shape 

X_test_input = Create_input_selected(len(sel_comb_acc), X_test, sel_comb_acc)
y_pred = knn_clf.predict(X_test_input)

# evaluate predictions
acc = accuracy_score(y_test, y_pred)
print("best accuracy on holdout, greedy search selected subset is", acc, file = writefile)
writefile.close()
