# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:00:33 2020

@author: kbaha
"""
from scipy.io import arff
import pandas as pd
import math
import csv
import os
import re
import numpy as np
from tslearn.utils import to_time_series_dataset

from sklearn.model_selection import train_test_split


def LoadPdData(filename_train, filename_test):
    data = arff.loadarff(filename_train)
    df = pd.DataFrame(data[0])

    return df

def LoadPdData_test(filename_test):
    data = arff.loadarff(filename_test)
    df = pd.DataFrame(data[0])
 
    return df

def list_files(directory):
    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files]
    return files
 
#files = list_files(r"C:\PhD\PhD\TSDatasets\Multivariate_arff\RacketSports\MTSDATA")

def FindUEAXy(files):
	'''
	Extract input as X and y from the UEA Time series arff datasets

	Parameters
	----------
	files : folder name of the files to be extracted

	Returns
	-------
	X_full : X in the 3D time series format
	y : labels

	'''
	df_comb = pd.DataFrame()
	for i in range(0,len(files)):
		# Load data
		data = arff.loadarff(files[i])
		df = pd.DataFrame(data[0])
		filename = files[i]
		
		target = list(dict(data[1]._attributes).values())[-1].values
			
		# rename the columns
		n_columns = len(df.columns)
		df.columns = range(n_columns)
		
		# find and store the variable the data belongs to
		var_group = re.findall("(\d+)_T", filename)[0]
		
		group_vect = np.full(len(df), var_group)
		df.insert(n_columns, "variable", group_vect)
		
		# Append the dataframes
		df_comb = df_comb.append(df, ignore_index = True)


	# split data into X, y, and groups
	X= df_comb.drop(df_comb.columns[n_columns-1:n_columns+1], axis=1)
	y= df_comb.iloc[:,n_columns-1].values
	var = df_comb.iloc[:,n_columns].values.astype(float)


	m_df, n_df = df_comb.shape

	# Find individual variable dataframes
	for i in range(1,len(np.unique(var))+1):
		
		if (len(np.unique(var))<10):
			j = i
		else:
			j = str(i).zfill(2)

		globals()['df%s' % j] = X[df_comb.variable=="%s" % j ]
		X_temp = globals()['df%s' % j]
		
		# Start reduce training size
		n, m = X_temp.shape
		y_true = y[0:n].astype(str)
		'''
		X_temp, X_discard, y_fin, y_discard = train_test_split(X_temp, y_true, test_size=0.4, random_state=42)
		# End reduce training size
		'''
		X_conv = to_time_series_dataset([X_temp])
		
		if i == 1:
			X_full = X_conv
		else:
			X_full = np.append(X_full, X_conv, axis=0)
			
	return X_full, y_true, target