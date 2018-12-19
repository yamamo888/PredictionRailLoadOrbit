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

import pdb

class Arima():#自己回帰
    def __init__(self,xData,tData):
        self.xData = xData
        self.tData = tData

        self.xDim = xData.shape[1]-1
        self.xNum = xData.shape[0]

        self.tNum = tData.shape[0]

        self.t = self.tData[self.tData['date'] == '2018-3-31']

        self.N = 10
        self.p = 10
        self.q = 12
        self.d = 1

        self.w_ar = np.random.normal(0.0, pow(100, -0.5), (self.p + 1, 1))
        self.w_ma = np.random.normal(0.0, pow(100, -0.5), (self.q + 1, 1))

        self.eps = np.array([np.random.normal(1,25) for _ in range(365*self.t.shape[0])])
        print(self.eps)

    def train(self):
        date_ar = []
        date_ma = []
        z_ar1 = np.empty(((self.N-self.d)*self.t.shape[0],0))
        z_ma1 = np.empty(((self.N-self.d)*self.t.shape[0],0))
        for i in range(self.p):
            z_ar0 = []
            for j in range(self.N-self.d):
                date_ar = np.append(date_ar, (self.t['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                z_ar0 = np.append(z_ar0, self.tData[self.tData['date'] == date_ar[-1]]['hll'])
            z_ar0 = z_ar0[np.newaxis].T
            z_ar1 = np.append(z_ar1, z_ar0,axis=1)

        for i in range(self.q):
            z_ma0 = []
            for j in range(self.N-self.d):
                z_ma0 = np.append(z_ma0, self.eps[(j+i+2)*self.t.shape[0]:(j+i+3)*self.t.shape[0]]) 
                #date_ma = np.append(date_ma, (self.t['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                #z_ma0 = np.append(z_ma0, self.tData[self.tData['date'] == date_ar[-1]]['hll'])
            z_ma0 = z_ma0[np.newaxis].T
            z_ma1 = np.append(z_ma1, z_ma0,axis=1)
        """
        for i in range(self.N):
            z_ar0 = []
            z_ma0 = []
            for j in range(self.p-self.d):
                date_ar = np.append(date_ar, (self.t['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                z_ar0 = np.append(z_ar0, self.tData[self.tData['date'] == date_ar[-1]]['hll'])
            z_ar0 = z_ar0[np.newaxis]
            z_ar1 = np.append(z_ar1, z_ar0,axis=0)

            for k in range(self.q-self.d):
                z_ma0 = np.append(z_ma0, self.eps[(k+i+2)*self.t.shape[0]:(k+i+3)*self.t.shape[0]]) 
                #pdb.set_trace()
                #date_ma = np.append(date_ma, (self.t['date'][-1:] - datetime.timedelta(days=k+i+2)).astype(str))
                #z_ma0 = np.append(z_ma0, self.tData[self.tData['date'] == date_ma[-1]]['hll'])
            #pdb.set_trace()
            z_ma0 = z_ma0[np.newaxis]
            z_ma1 = np.append(z_ma1, z_ma0,axis=0)
        """    
        z_ar1 = np.append(z_ar1, np.ones([z_ar1.shape[0],1]),axis=1)
        z_ma1 = np.append(z_ma1, np.ones([z_ma1.shape[0],1]),axis=1)

        y = []
        date_y = []
        e = []
        for i in range(self.N-self.d):
            date_y = np.append(date_y, (self.t['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
            y = np.append(y, self.tData[self.tData['date'] == date_y[-1]]['hll'])
            e = np.append(e, self.eps[i*self.t.shape[0]:(i+1)*self.t.shape[0]])
        y = y[np.newaxis].T
        e = e[np.newaxis].T

        sigma_ar0 = np.matmul(z_ar1.T, z_ar1)
        sigma_ar1 = np.matmul(z_ar1.T, y)
        self.w_ar = np.matmul(sigma_ar0, sigma_ar1)
        print(self.w_ar)
        pdb.set_trace()

        sigma_ma0 = np.matmul(z_ma1.T, z_ma1)
        sigma_ma1 = np.matmul(z_ma1.T, e)
        self.w_ma = np.matmul(sigma_ma0, sigma_ma1)
        print(self.w_ma)

    def predict(self,t):
        #pdb.set_trace()
        date = []
        y = []
        for i in range(self.p):
            date = np.append(date, (t['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
            y = np.append(y, self.tData[self.tData['date'] == date[-1]]['hll'])
        y = y.reshape([self.p,t.shape[0]])

        y = self.w_ar[0] + np.matmul(self.w_ar[1:].T, y) + np.matmul(self.w_ma, self.eps[1:self.p]) + self.eps[0]
        #y = self.w_ar[0] + np.matmul(self.w_ar[1:].T, y) + self.eps[0]
        return y

    def loss(self,tDate):
        t = np.array(tDate['hll'])[np.newaxis]
        #pdb.set_trace()
        #t = t[t['date'] == '2018-03-31']
        num = pow(t - self.predict(tDate),2)
        loss = np.sum(num) / (t.shape[1])
        return loss

class trackData():
    def __init__(self):#trainの読み込み

        self.train_xData = []
        # self.test_xData = []
        self.train_tData = []
        # self.test_tData = []

        fileind = ['A','B','C','D']

        for no in range(len(fileind)):
            fname_xTra = "xTrain_{}.binaryfile".format(fileind[no])
            # fname_xTes = "xTest_{}.binaryfile".format(fileind[no])
            fname_tTra = "tTrain_{}.binaryfile".format(fileind[no])
            # fname_tTes = "tTest_{}.binaryfile".format(fileind[no])

            self.load_file(fname_xTra,self.train_xData)
            # self.load_file(fname_xTes,self.test_xData)
            self.load_file(fname_tTra,self.train_tData)
            # self.load_file(fname_tTes,self.test_tData)


    def load_file(self,filename,data):
        f = open(filename,'rb')
        data.append(pickle.load(f))
        f.close


if __name__ == "__main__":

    isWindows = False

    fileind = ['A','B','C','D']

    mytrackData = trackData()
    #trackData.xData = [xData_A,xData_B,xData_C,xData_D]
    #trackData.tData = [tData_A,tData_B,tData_C,tData_D]
    # ar_A = Ar(mytrackData.xData[0],mytrackData.tData[0])
    # T = tData[tData['date'] == '2018-03-31']
    # w_A = ar_A.train()

    # ar_list = []
    w_list = []

    for no in range(len(fileind)):
        arima = Arima(mytrackData.train_xData[0],mytrackData.train_tData[0])
        # ar_list.append(ar)
        arima.train()
        w_list.append(arima.w_ar)

    f = open("w_list.binaryfile","wb")
    pickle.dump(w_list,f)
    f.close
