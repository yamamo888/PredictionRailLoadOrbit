#-*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:25:31 2018

@author: yu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:42:54 2018

@author: yu
"""

import numpy as np
import pandas as pd
#import tensorflow as tf
import os
import datetime as dt
import heapq
import pickle
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pylab as plt

import datetime
import time

import pdb

import concurrent.futures

import xgboost as xgb

#------------------------------------------------------------
# xgboostの実装
class xgboost():
	#------------------------------------------------------------
	def __init__(self, xData, yData):
		self.xData = xData
		self.yData = yData

		self.xDim = xData.shape[1]-1
		self.xNum = xData.shape[0]

		self.yNum = yData.shape[0]
	#------------------------------------------------------------

	#------------------------------------------------------------
	def learn(self):
		# xgboostモデルの作成
		mod = xgb.XGBRegressor()
		mod.fit(self.xData, self.yData)

		t = np.array(tDate['hll'])[np.newaxis]
		#pdb.set_trace()
		num = pow(t - self.predict(tDate),2)
		loss = np.sum(num) / (t.shape[1])
		return loss
	#------------------------------------------------------------

#------------------------------------------------------------

#------------------------------------------------------------

class trackData():
	dataPath = '../data'
	def __init__(self,p):#trainの読み込み

		# self.train_xData = []
		self.train_eData = []
		self.train_tData = []
		# self.test_tData = []

		fileind = ['A','B','C','D']

		for no in range(len(fileind)):
			# fname_xTra = "track_xTrain_{}.binaryfile".format(fileind[no])
			fname_eTra = "equipment_eTrain_{}.binaryfile".format(fileind[no])
			fname_tTra = "track_tTrain_{}.binaryfile".format(fileind[no])
			# fname_tTes = "tTest_{}.binaryfile".format(fileind[no])
			self.load_file(fname_eTra,self.train_eData)
			# self.load_file(fname_xTes,self.test_xData)
			self.load_file(fname_tTra,self.train_tData)
			# self.load_file(fname_tTes,self.test_tData)
		self.make_trainData(p)


	def load_file(self,filename,data):
		fullpath = os.path.join(self.dataPath, filename)
		f = open(fullpath,'rb')
		data.append(pickle.load(f))
		f.close

	def make_trainData(self,p):
		self.train_xData = []
		self.train_yData = []
		fileind = ['A','B','C','D']

		for no in range(len(fileind)):
			t = self.train_tData[no]
			krage = t.shape[1]
			xdata = []
			ydata = []
			for i in range(krage):
				t_kr = t[:][i]
				ydata.append(t_kr[p:])
				x = np.array([t_kr[:p]])
				pdb.set_trace()
				for j in range(krage-p-1):
					x = np.append(x,t_kr[j+1:j+p+1],axis = 0)
				xdata.append(x)
			self.train_xData.append(xdata)
			self.train_yData.append(ydata)

#------------------------------------------------------------

#------------------------------------------------------------
if __name__ == "__main__":

	isWindows = False

	fileind = ['A','B','C','D']

	day = 5 #さかのぼる日数
	myData = trackData(day)
	start_all = time.time()
	#----------------------------------------------
	result = []
	for no in range(len(fileind)):
		w = xgboost(myData.train_xData[no],myData.train_yData[no])
		result.append(w)

	end_time = time.time() - start_all
	print("time : {0}".format(end_time) + "[sec]")
	f_xgb = open("xgb_w_list.binaryfile","wb")
	pickle.dump(result,f_xgb)
	f_xgb.close()
