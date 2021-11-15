# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:19:45 2020

@author: kbaha
"""
import itertools 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import math
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import time

def Calc_MIScore(var_no, y_predicted_label, i, combs):
    MI_score_ins = 0
    comb_ind = list(itertools.combinations(combs[i], 2))
    for j in range(0,len(comb_ind)):
        first = comb_ind[j][0]-1
        second = comb_ind[j][1]-1
        MI_score_ins = MI_score_ins + adjusted_mutual_info_score(y_predicted_label[first],y_predicted_label[second])
    return MI_score_ins/len(comb_ind)
 
def find_dtw_dist(combinations, dist):
    dist_total = 0
    for j in range(0,len(combinations)):
        dist_total = dist[combinations[j]-1] + dist_total
    return dist_total/len(combinations)

def knn_clfcrossvalxv(X,y):
    xv = []
    reps = 1

    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")

    for i in range(reps):
        scores = cross_val_score(knn_clf, X, y, cv=5)
        xv = scores.mean()
    return xv

def knn_clfcrossvalxv_precomp(X,y):
    xv = []
    reps = 1

    knn_clf = KNeighborsClassifier(n_neighbors=1, metric="precomputed")

    for i in range(reps):
        scores = cross_val_score(knn_clf, X, y, cv=5, scoring = 'accuracy')
        xv = scores.mean()
    return xv

def Create_input(var_no, X_full, i, combs):
    X_first = X_full[combs[i][0]-1,:,:]
    for j in range(0,var_no-1):
        second= combs[i][j+1]-1
        X_input = np.append(np.atleast_3d(X_first),np.atleast_3d(X_full[second,:,:]), axis = 2)
        X_first = X_input
    return X_input
 
def Create_input_selected(var_no, X_full, sel_comb):
    X_first = X_full[sel_comb[0]-1,:,:]
    for j in range(0,var_no-1):
        second= sel_comb[j+1]-1
        X_input = np.append(np.atleast_3d(X_first),np.atleast_3d(X_full[second,:,:]), axis = 2)
        X_first = X_input
    return X_input
 
def Cal_numer(MI_y, combs, i, var_no):
    MI_num = 0
    for j in range(0,var_no):
        first = combs[i][j]-1
        MI_num = MI_num + MI_y.values[first]
    return ((var_no*(MI_num/var_no)))
 
def FindMerit_Score(MI_y,var_no, n, sel_comb, y_predicted_label, X_full):
	'''
	

	Parameters
	----------
	MI_y : Single feature MI.
	var_no : Feature subset size.
	n : total number of variables being considered.
	sel_comb : The selected feature subset in the previous step. Can be empty when var_no is 2

	Returns
	-------
	None.


	''' 

	# Create a consecutive list with all the variables
	var_list = list(range(1,n+1))

	# Find the unique  combinations of variables
	if (var_no == 2):
		combs = list(itertools.combinations(var_list, var_no))
	else:
		j=0
		combs = []
		for i in sel_comb:
			var_list.remove(i)
		for i in var_list:
			combs.insert(j, sel_comb + (i,)) 
			j=j+1
			  
	# Find the Mutual information between y labels
	MI_score = []
	for i in range(0,len(combs)):
		
		MI_score.append(Calc_MIScore(var_no, y_predicted_label, i, combs))

	merit_score = []
	for i in range(0,len(combs)):

		MS = (Cal_numer(MI_y, combs, i, var_no))/(math.sqrt(var_no+var_no*(var_no-1)*(MI_score[i])))
		merit_score.append(MS[0])
	
	return merit_score, combs

	

def FindCombsaccuracy(var_no, combs, dist, y_true):
	accuracy = []
	
	for i in range(0,len(combs)):
		#X_input = Create_input(var_no, X_full, i, combs)
		input_dist = find_dtw_dist(combs[i],dist)
		accuracy.append(knn_clfcrossvalxv_precomp(input_dist,y_true))
	
	return accuracy

def Findaccuracy(var_no, n, sel_comb, X_full, dist, y_true):
	
	
	# Create a consecutive list with all the variables
	var_list = list(range(1,n+1))

	# Find the unique  combinations of variables
	if (var_no == 2):
		combs = list(itertools.combinations(var_list, var_no))
	else:
		j=0
		combs = []
		for i in sel_comb:
			var_list.remove(i)
		for i in var_list:
			combs.insert(j, sel_comb + (i,)) 
			j=j+1

	accuracy = []

	for i in range(0,len(combs)):
		#X_input = Create_input(var_no, X_full, i, combs)
		input_dist = find_dtw_dist(combs[i],dist)
		accuracy.append(knn_clfcrossvalxv_precomp(input_dist,y_true))
	
	
	
	return accuracy, combs