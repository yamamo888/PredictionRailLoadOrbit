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

class Ar():#自己回帰
    def __init__(self,x,t):
        self.x = x
        self.t = t

        self.xDim = x.shape[1]-1
        self.xNum = x.shape[0]

        self.tNum = t.shape[0]

        self.N = 50
        self.p = 50

        self.w = np.random.normal(0.0, pow(100, -0.5), (self.p + 1, 1))

    def train(self):
        t = self.t[self.t['date'] == '2018-03-31']
        date = []
        z1 = np.empty((self.N*t.shape[0],0))
        for i in range(self.p):
            z0 = []
            for j in range(self.N):
                date = np.append(date, (t['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                z0 = np.append(z0, self.t[self.t['date'] == date[-1]]['hll'])
            #pdb.set_trace()
            z0 = z0[np.newaxis].T
            z1 = np.append(z1, z0,axis=1)
        z1 = np.append(z1, np.ones([z1.shape[0],1]),axis=1)

        y = []
        for i in range(self.N):
            date = np.append(date, (t['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
            y = np.append(y, self.t[self.t['date'] == date[-1]]['hll'])
        y = y[np.newaxis].T

        sigma0 = np.matmul(z1.T, z1)
        sigma1 = np.matmul(z1.T, y)
        self.w = np.matmul(sigma0, sigma1)
        #pdb.set_trace()

    def predict(self,t):
        #pdb.set_trace()
        date = []
        y = []
        for i in range(self.p):
            date = np.append(date, (t['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
            y = np.append(y, self.t[self.t['date'] == date[-1]]['hll'])
        y = y.reshape([self.p,t.shape[0]])

        #print("date :\n", date)
        #print("y :\n", y)
        #pdb.set_trace()

        y = self.w[0] + np.matmul(self.w[1:].T, y)
        return y

    def loss(self,tDate):
        t = np.array(tDate['hll'])[np.newaxis]
        pdb.set_trace()
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
        ar = Ar(mytrackData.train_xData[0],mytrackData.train_tData[0])
        # ar_list.append(ar)
        ar.train()
        w_list.append(ar.w)

    f = open("w_list.binaryfile","wb")
    pickle.dump(w_list,f)
    f.close
