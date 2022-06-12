# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:55:57 2022

@author: kbaha
"""
import numpy as np
import re


def ReadDTW(Dataset_name):

	#with open('C:\PhD\PhD\MSTS_Final\Results\\' + Dataset_name + "_DTW.txt") as f:
	with open('Results/' + Dataset_name + "_DTW.txt") as f:
		lines = f.readlines()
		
	array3D = []
	newline = [w.replace("\n", " ") for w in lines]

	onestring = ' '.join([str(item) for item in newline])


	start = 'array(['
	end = '])'
	firstsplit = (onestring.split(start))

	for i in range(0,len(firstsplit)-1):
		ss = (firstsplit[i+1].split(end)[0])

		print(len(ss))
	
	
		start1 = '['
		end1 = ']'
	
		secondsplit = (ss.split(start1))
	
		for j in range(0,len(secondsplit)-1):
			s3 = (secondsplit[j+1].split(end1)[0])
		
			if j==0:
				array2D = np.fromstring(s3, dtype=float, sep=',')
				array2D = array2D.reshape(len(array2D),1)
			else:
				array2D = np.insert(array2D, j, np.fromstring(s3, dtype=float, sep=','), axis = 1)
		
	
		m,n = array2D.shape
		if i == 0:
		
			#array3D = np.reshape(array2D, (m,n,1))
			array3D.insert(i, array2D)
	
		else:
			#array3D = np.insert(array3D, i, array2D, axis = 2)
			array3D.insert(i, array2D)

	dist = array3D
	
	return dist


