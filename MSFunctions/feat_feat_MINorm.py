# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:06:11 2022

@author: kbaha
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:44:44 2022

@author: kbaha
"""

from MSFunctions.Load_Data import FindUEAXy, list_files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from tslearn.metrics import dtw, cdist_dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from sklearn.neighbors import NearestNeighbors

from scipy.special import digamma
#from MI_knn import kraskov_mi
from sklearn.neighbors import NearestNeighbors

def I_gain_calc(k,m,I_gain_sum):
	I_gain = digamma(k) + digamma(m) - (1/m)*I_gain_sum
	
	return I_gain

k = 5 # The k in use in k-nn

Dataset_Name = 'JapaneseVowels'

# Set the name of the Dataset
files = list_files(r"C:\PhD\PhD\TSDatasets\Multivariate_arff\\"+Dataset_Name+"\MTSDATA\Train")
files_test = list_files(r"C:\PhD\PhD\TSDatasets\Multivariate_arff\\"+Dataset_Name+"\MTSDATA\Test")

X_full, y_true, target = FindUEAXy(files)

# Find size of X
n,m,l = X_full.shape

class_label, class_count = np.unique(y_true, return_counts = True)


dist = []
for i in range(0,n):
	dist.append(cdist_dtw(X_full[i,:,:], X_full[i,:,:]))

I_gain_sum = 0

# Encode the labels
labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y_true)
	
#%%
#knn = KNeighborsTimeSeries(n_neighbors=k, metric="dtw")

def knnMI_FF(dist, k, n, y_true, m):
	
	'''
	This function calculates feature-feature correlation using k-nn approach
	Inputs:
		dist - DTW distance matri 
		k - k in k-nn
		n - number of features
		m - number of instances
	Output:
		I_gain - information gain between pairs of features
	'''
	I_gain = pd.DataFrame()
	V_ts1_full = pd.DataFrame()
	V_ts2_full = pd.DataFrame()
	
	# iterating over each feature
	for ts_i in range(0,n):
		for ts_j in range(0,n):
		
			# Combine the two features to Find the multivariate feature matrix
			#X_mv = np.append(np.atleast_3d(X_full[ts_i,:,:]),np.atleast_3d(X_full[ts_j,:,:]), axis= 2)
			I_gain_sum = 0
			V_ts1_sum = 0
			V_ts2_sum = 0
		
			# Find chebyshev for the multivariate two feature combination
			chebyshev_dist = np.maximum(dist[ts_i][:], dist[ts_j][:])
		
			knn = NearestNeighbors(n_neighbors=k+1, metric = 'precomputed')
			# find first kth neighbour from instance using multivaritate point
			knn.fit(chebyshev_dist, y_true)
			dists_1, ind_1 = knn.kneighbors(chebyshev_dist, return_distance = True)

			# Iterate over all the instances and sum the I_gain for each instance
			for j in range(0,m):

				V_ts1 = (dist[ts_i][:][j]<=dists_1[j,k]).sum()
				V_ts2 = (dist[ts_j][:][j]<=dists_1[j,k]).sum()
	
				I_gain_sum = I_gain_sum + (digamma(V_ts1) + digamma(V_ts2))
				V_ts1_sum = V_ts1_sum + V_ts1
				V_ts2_sum = V_ts2_sum + V_ts2
	
			I_gain.loc[ts_i,ts_j] = I_gain_calc(k,m,I_gain_sum)
			V_ts1_full.loc[ts_i, ts_j] = V_ts1_sum
			V_ts2_full.loc[ts_i, ts_j] = V_ts2_sum
		
	return I_gain, V_ts1_full, V_ts2_full


I_gain, V_ts1_full, V_ts2_full = knnMI_FF(dist, k, n, y_true, m)

ratio_V = V_ts1_full-V_ts2_full

np.fill_diagonal(I_gain.values, 'NaN')
np.fill_diagonal(ratio_V.values, 'NaN')
plt.figure(dpi = 300)
plt.scatter(I_gain, ratio_V)

plt.xlabel("Information Gain")
plt.ylabel("V_ts1 - V_ts2 (sum of all instances)")

'''
import seaborn as sns
import matplotlib.pyplot as plt
f2 = plt.figure(dpi = 300)
ay = sns.heatmap(I_gain, cmap = "Reds")
ay.set_ylabel("Features")
ay.set_xlabel("Features")

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

data_classes = target
y_pred = pd.DataFrame()
acc = []

for i in range(0,n):
	
	knn_clf = KNeighborsClassifier(n_neighbors=1, metric = 'precomputed')
	y_prediction = cross_val_predict(knn_clf, dist[i],y_true, cv=10)
	y_pred.insert(i,i,y_prediction)
	 
	y_acc = cross_val_score(knn_clf, dist[i],y_true, cv=10, scoring = 'accuracy')
	acc.append(y_acc.mean())
 

# MI for each variable compared with true value
y_predicted_label = pd.DataFrame()
MI_y = pd.DataFrame()
for i in range(0,n):
    y_predicted_label.insert(i,i,y_pred[i].apply(data_classes.index))
    MI_y.loc[i,0] = adjusted_mutual_info_score(y_pred[i], y_true)
	 
# MI for each variable compared with true value

MI_xy = pd.DataFrame()
for i in range(0,n):
    y_predicted_label = pd.DataFrame()
    for j in range(0,n):
        y_predicted_label.insert(j,j,y_pred[j].apply(data_classes.index))
        MI_xy.loc[i,j] = adjusted_mutual_info_score(y_pred[i], y_pred[j])
#%%
from scipy.stats import pearsonr
np.fill_diagonal(I_gain.values, 'NaN')
np.fill_diagonal(MI_xy.values, 'NaN')
#np.fill_diagonal(I_gain_kraskow.values, 'NaN')

plt.figure(dpi = 300)
plt.scatter(I_gain, MI_xy)
plt.xlabel("knn technique")
plt.ylabel("our technique")
plt.title(Dataset_Name + " - Feature Feature correlation score")
#plt.xlim(0.0,0.2)
#plt.ylim(0.8,3.5)
corr = pearsonr(I_gain.values.reshape(n*n)[~np.isnan(I_gain.values.reshape(n*n))], MI_xy.values.reshape(n*n)[~np.isnan(MI_xy.values.reshape(n*n))])
plt.text(1.0, 0.26, "r = " + str(round(corr[0],2)), fontsize=12, style='italic')
'''