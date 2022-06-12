# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:13:05 2022

@author: kbaha
"""

from MSFunctions.Load_Data import FindUEAXy, list_files
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tslearn.metrics import dtw, cdist_dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.special import digamma

def I_gain_calc(k,m,I_gain_sum):
	I_gain = digamma(k) + digamma(m) - (1/m)*I_gain_sum
	
	return I_gain

I_gain_cc = pd.DataFrame()
k = 5 # The k in use in k-nn

Dataset_Name = 'RacketSports'

# Set the name of the Dataset
files = list_files(r"C:\PhD\PhD\TSDatasets\Multivariate_arff\\"+Dataset_Name+"\MTSDATA\Train")
files_test = list_files(r"C:\PhD\PhD\TSDatasets\Multivariate_arff\\"+Dataset_Name+"\MTSDATA\Test")

X_full, y_true, target = FindUEAXy(files)

# Find size of X
n,m,l = X_full.shape


def knnMI_FC(X_full, y_true, k, n, m):
	
	# Encode the labels
	labelencoder = LabelEncoder()
	y_encoded = labelencoder.fit_transform(y_true)
	# Find the count of unique classes
	class_label, class_count = np.unique(y_encoded, return_counts = True)
	
	# Iterate for each class type to find distances within class

	for feat in range(0,n):

		I_gain_sum = 0
		for label_type in class_label:
			index_label = np.where(y_encoded == label_type)
			X_singleclass = X_full[feat,np.where(y_encoded == label_type)[0],:]
			y_singleclass = y_encoded[np.where(y_encoded == label_type)]
			knn = KNeighborsTimeSeries(n_neighbors=k, metric="dtw")

			# fit to single class data
			knn.fit(X_singleclass, y_singleclass)
			dists, ind = knn.kneighbors(X_singleclass, return_distance = True)

		dist_all = []

		dist_mat = cdist_dtw(X_full[feat,:,:], X_full[feat,:,:])

		# Iterate for each class type to find distances within class
		I_gain_sum = 0
		for label_type in class_label:
	
			index_label = np.where(y_encoded == label_type)
			X_singleclass = X_full[feat,np.where(y_encoded == label_type)[0],:]
			y_singleclass = y_encoded[np.where(y_encoded == label_type)]
			knn = KNeighborsTimeSeries(n_neighbors=k, metric="dtw")

			# fit to single class data
			knn.fit(X_singleclass, y_singleclass)
			dists, ind = knn.kneighbors(X_singleclass, return_distance = True)
	
			for j in range(0,len(dists)):
				V_tsp = [(dist_mat[index_label[0][j]]<=dists[j, k-1]).sum()]
		
				I_gain_sum = I_gain_sum + (digamma(class_count[label_type]) + digamma(V_tsp))
	
		I_gain_cc.loc[feat,0] = I_gain_calc(k,m,I_gain_sum)[0] 
		
	return I_gain_cc

I_gain_cc = knnMI_FC(X_full, y_true, k, n, m)

	
	
#Usage Functions
def kraskov_mi_xy(x,y,k=5):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using KSG mutual information estimator
		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter
		Output: one number of I(X;Y)
	'''

	assert len(x)==len(y), "Lists should have same length"
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	dx = len(x[0])   	
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans_xy = -digamma(k) + digamma(N) + (dx+dy)*log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
	ans_x = digamma(N) + dx*log(2)
	ans_y = digamma(N) + dy*log(2)
	for i in range(N):
		ans_xy += (dx+dy)*log(knn_dis[i])/N
		ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*log(knn_dis[i])/N
		ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*log(knn_dis[i])/N
		
	return ans_x+ans_y-ans_xy
