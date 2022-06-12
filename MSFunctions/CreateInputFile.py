# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:43:19 2020

@author: kbaha
"""

import numpy as np
import pandas as pd
from ButterworthFilter import butter_lowpass, butter_lowpass_filter
from Plot_EachSignal_added import filter_series, FindSegmentedPoints
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt

def ReadXandy(filename):
	'''
	Parameters
	----------
	filename : The filename witht the time series and labels

	Returns
	-------
	filtered_series : the time series
	y : the valgus/normal label
	y_dom : dominant or non-dominant label
	y_sub : subject number label

	'''
	data_P2 = pd.read_csv(filename)
	X= data_P2.drop(data_P2.columns[0], axis=1)
	X= X.drop(X.columns[0], axis=1)
	X= X.drop(X.columns[0], axis=1)
	X_P2=X.values
	y= data_P2.iloc[:,1]
	y_dom= data_P2.iloc[:,2]
	y_sub = data_P2.iloc[:,0]
	
	# Filter requirements.
	order = 8
	fs = 300.0 # sample rate, Hz
	cutoff = 20 # desired cutoff frequency of the filter, Hz
	filtered_series = butter_lowpass_filter(X_P2,cutoff,fs,order)
	
	return filtered_series, y, y_dom, y_sub

def InsertSections(filtered_series_L, filtered_series_R, Segment_L, Segment_R, y_LS, y_RS):
	# This function outputs the X for each segment of the lunge and requires the 
	# left and right time series seperately inputted as well as the left and right 
	# segments and the labels
	
	y_full = pd.DataFrame()
	num = 0
	ts_S1 = pd.DataFrame()
	ts_S2 = pd.DataFrame()
	ts_S3 = pd.DataFrame()
	ts_S12 = pd.DataFrame()

	for i in range(0,len(filtered_series_L)):
		if y_LS[i] == 0 or y_LS[i] ==1:
			#Insert the time series for the desired segment of the lunge
			ts_S1.insert(num,num,pd.Series(filtered_series_L[i][Segment_L['IC'][i]:Segment_L['IC1'][i]]))
			ts_S2.insert(num,num,pd.Series(filtered_series_L[i][Segment_L['IC1'][i]:Segment_L['CD'][i]]))
			ts_S3.insert(num,num,pd.Series(filtered_series_L[i][Segment_L['CD'][i]:Segment_L['IC2'][i]]))
			ts_S12.insert(num,num,pd.Series(filtered_series_L[i][Segment_L['SMid'][i]:Segment_L['CD'][i]]))
		
			y_full.insert(num,num,[y_LS[i]])

			num = num+1
			
		if y_RS[i] == 0 or y_RS[i]==1:
		
			# Insert Right data
			ts_S1.insert(num,num,pd.Series(filtered_series_R[i][Segment_R['IC'][i]:Segment_R['IC1'][i]]))
			ts_S2.insert(num,num,pd.Series(filtered_series_R[i][Segment_R['IC1'][i]:Segment_R['CD'][i]]))
			ts_S3.insert(num,num,pd.Series(filtered_series_R[i][Segment_R['CD'][i]:Segment_R['IC2'][i]]))
			ts_S12.insert(num,num,pd.Series(filtered_series_R[i][Segment_R['SMid'][i]:Segment_R['CD'][i]]))
			
			y_full.insert(num,num,[y_RS[i]])

			num = num+1
			
	#ts_S1 = to_time_series_dataset(ts_S1.T.values)
	#ts_S2 = to_time_series_dataset(ts_S2.T.values)
	#ts_S3 = to_time_series_dataset(ts_S3.T.values)
			
	#X_Out = np.concatenate([ts_S1,ts_S2, ts_S3], axis=2)
	X_Out = to_time_series_dataset([ts_S1.values,ts_S2.values,ts_S3.values,ts_S12.values])
	#y_full = y_full.replace(2,1)
	return X_Out, y_full
		
def OutputJerk(filtered_series_L, filtered_series_R, Segment_L, Segment_R, y_LS, y_RS):
	# Specialised version of the InsertSections function for the jerk
	num = 0
	ts_jerk_S1 = pd.DataFrame()
	ts_jerk_S2 = pd.DataFrame()
	ts_jerk_S3 = pd.DataFrame()
	ts_jerk_S12 = pd.DataFrame()
	
	for i in range(0,len(filtered_series_L)):
		
		if y_LS[i] == 0 or y_LS[i] ==1:
			
			# Insert Left Jerk data
			ts_int = pd.Series(filtered_series_L[i][Segment_R['IC'][i]:Segment_R['IC1'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S1.insert(num,num,slope)
			
			ts_int = pd.Series(filtered_series_L[i][Segment_R['IC1'][i]:Segment_R['CD'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S2.insert(num,num,slope)
			
			ts_int = pd.Series(filtered_series_L[i][Segment_R['CD'][i]:Segment_R['IC2'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S3.insert(num,num,slope)
			
			ts_int = pd.Series(filtered_series_L[i][Segment_R['SMid'][i]:Segment_R['CD'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S12.insert(num,num,slope)
			
			num = num+1
			
		if y_RS[i] == 0 or y_RS[i]==1:
			
			ts_int = pd.Series(filtered_series_R[i][Segment_R['IC'][i]:Segment_R['IC1'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S1.insert(num,num,slope)
			
			ts_int = pd.Series(filtered_series_R[i][Segment_R['IC1'][i]:Segment_R['CD'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S2.insert(num,num,slope)
			
			ts_int = pd.Series(filtered_series_R[i][Segment_R['CD'][i]:Segment_R['IC2'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S3.insert(num,num,slope)
			
			ts_int = pd.Series(filtered_series_R[i][Segment_R['SMid'][i]:Segment_R['CD'][i]])
			slope = pd.Series(np.gradient(ts_int), ts_int.index, name='slope')
			ts_jerk_S12.insert(num,num,slope)
			num = num+1
	
	X_Out = to_time_series_dataset([ts_jerk_S1.values,ts_jerk_S2.values,ts_jerk_S3.values,ts_jerk_S12.values])
	
	return X_Out

#%% Import needed files
#Import x acc data
def CreateXy():
	filename_LeftShank= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_X_L.csv"
	filename_RightShank = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_X_R.csv"

	filtered_LeftShank_X, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank)
	filtered_RightShank_X, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank)

	#Import y acc data
	filename_LeftShank= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_Y_L.csv"
	filename_RightShank = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_Y_R.csv"

	filtered_LeftShank_Y, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank)
	filtered_RightShank_Y, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank)

	#Import Z acc data
	filename_LeftShank= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_Z_L.csv"
	filename_RightShank = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_Z_R.csv"

	filtered_LeftShank_Z, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank)
	filtered_RightShank_Z, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank)

	#Import X gyro data
	filename_LeftShank= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_L_X.csv"
	filename_RightShank = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_R_X.csv"

	filtered_LeftShank_X_gyro, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank)
	filtered_RightShank_X_gyro, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank)

	#Import Y gyro data
	filename_LeftShank= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_L_Y.csv"
	filename_RightShank = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_R_Y.csv"

	filtered_LeftShank_Y_gyro, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank)
	filtered_RightShank_Y_gyro, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank)

	#Import Z gyro data
	filename_LeftShank= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_L_Z.csv"
	filename_RightShank = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_R_Z.csv"

	filtered_LeftShank_Z_gyro, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank)
	filtered_RightShank_Z_gyro, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank)

	#Import files needed for segmentation
	filename_L_gyro= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_L_X_segment.csv"
	filename_L = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_L_segment.csv"
	Segment_L = FindSegmentedPoints(filename_L_gyro, filename_L)

	filename_R_gyro= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\Gyroshank_R_X_segment.csv"
	filename_R = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_R_segment.csv"
	Segment_R = FindSegmentedPoints(filename_R_gyro, filename_R)
	
	filename_LeftShank_mag= "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_L.csv"
	filename_RightShank_mag = "C:\PhD\PhD\SymmetryWork\\asym_lunge_trials\input_files\shank_R.csv"
	filtered_LeftShank_rawmag, y_LS, y_dom_L, y_sub_L = ReadXandy(filename_LeftShank_mag)
	filtered_RightShank_rawmag, y_RS, y_dom_R, y_sub_R = ReadXandy(filename_RightShank_mag)
			
	magnitude_Left = pd.DataFrame()
	for j in range(0,len(filtered_LeftShank_X.T)):
		mag = np.sqrt(filtered_LeftShank_X.T[j]**2 + filtered_LeftShank_Y.T[j]**2 + filtered_LeftShank_Z.T[j]**2)
		magnitude_Left.insert(j,j, mag)
	
	magnitude_Right = pd.DataFrame()
	for j in range(0,len(filtered_RightShank_X.T)):
		mag = np.sqrt(filtered_RightShank_X.T[j]**2 + filtered_RightShank_Y.T[j]**2 + filtered_RightShank_Z.T[j]**2)
		magnitude_Right.insert(j,j, mag)
	#%%
	#acceleration data
	X_ts_X, y_full = InsertSections(filtered_LeftShank_X, filtered_RightShank_X, Segment_L, Segment_R, y_LS, y_RS)
	X_ts_Y, y_full = InsertSections(filtered_LeftShank_Y, filtered_RightShank_Y, Segment_L, Segment_R, y_LS, y_RS)
	X_ts_Z, y_full = InsertSections(filtered_LeftShank_Z, filtered_RightShank_Z, Segment_L, Segment_R, y_LS, y_RS)

	X_ts_G_X, y_full = InsertSections(filtered_LeftShank_X_gyro, filtered_RightShank_X_gyro, Segment_L, Segment_R, y_LS, y_RS)
	X_ts_G_Y, y_full = InsertSections(filtered_LeftShank_Y_gyro, filtered_RightShank_Y_gyro, Segment_L, Segment_R, y_LS, y_RS)
	X_ts_G_Z, y_full = InsertSections(filtered_LeftShank_Z_gyro, filtered_RightShank_Z_gyro, Segment_L, Segment_R, y_LS, y_RS)

	#X_ts_J_X = OutputJerk(filtered_LeftShank_X, filtered_RightShank_X, Segment_L, Segment_R, y_LS, y_RS)
	#X_ts_J_Y = OutputJerk(filtered_LeftShank_Y, filtered_RightShank_Y, Segment_L, Segment_R, y_LS, y_RS)
	#X_ts_J_Z = OutputJerk(filtered_LeftShank_Z, filtered_RightShank_Z, Segment_L, Segment_R, y_LS, y_RS)
	print(filtered_LeftShank_rawmag.dtype)
	print(magnitude_Left.values.dtype)

	X_ts_rawmag, y_full = InsertSections(filtered_LeftShank_rawmag, filtered_RightShank_rawmag, Segment_L, Segment_R, y_LS, y_RS)
	X_ts_filtmag, y_full = InsertSections(magnitude_Left.values, magnitude_Right.values, Segment_L, Segment_R, y_LS, y_RS)

	#%%		
	
	y_full = y_full.values.reshape(len(y_full.T),)
	#X_full = np.concatenate([X_ts_X.T,X_ts_Y.T, X_ts_Z.T, X_ts_G_X.T, X_ts_G_Y.T, X_ts_G_Z.T, X_ts_J_X.T, X_ts_J_Y.T, X_ts_J_Z.T], axis=2)
	X_full = np.concatenate([X_ts_X.T,X_ts_Y.T, X_ts_Z.T, X_ts_G_X.T, X_ts_G_Y.T, X_ts_G_Z.T, X_ts_rawmag.T, X_ts_filtmag.T], axis=2)
	l,m,n = X_full.shape
	
	plt.figure(dpi = 300)
	plt.plot(filtered_RightShank_X_gyro[10])
	plt.title('X Gyro')
	plt.axvline(x=Segment_R['IC'][10], c='k')
	plt.axvline(x=Segment_R['IC1'][10], c='k')
	plt.axvline(x=Segment_R['CD'][10], c='k')
	plt.axvline(x=Segment_R['IC2'][10], c='k')
	plt.grid(which = 'major')

	return X_full, y_full