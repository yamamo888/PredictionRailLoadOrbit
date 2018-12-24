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

#------------------------------------------------------------
# ARIMAモデルでの自己回帰 :
#   ARモデルとMAモデルを
#   htitps://k-san.link/ar-process/を参照
#
# self.N    : 使用する日数
# self.p    : 遡る日数(ARモデル)
# self.q    : 遡る日数(MAモデル)
# self.d    : d次階差時系列
# self.w_ar : ARモデルでのパラメータ(重み)
# self.w_ma : MAモデルでのパラメータ(重み)
# self.eps  : MAモデルで使用するホワイトノイズ
#------------------------------------------------------------
class Arima():
    def __init__(self,xData,tData):
        self.xData = xData
        self.tData = tData

        self.xDim = xData.shape[1]-1
        self.xNum = xData.shape[0]

        self.tNum = tData.shape[0]

        self.t = self.tData[self.tData['date'] == '2018-3-31']
        self.kDate = np.empty([365, 3]).tolist()
        self.N = 10
        self.p = 10
        self.q = 12
        self.d = 1

        self.w_ar = []
        self.w_ma = []        

        self.eps = [np.random.normal(1,25) for _ in range(365)]
     
    #------------------------------------------------------------
    # ARモデル :
    #   時系列データの目的変数のみで将来の予測を行う回帰
    #   self.N日のデータからself.p日遡って回帰を行う
    #
    # z_ar0     : self.w_arを求めるためのZ行列の列の要素
    # z_ar1     : self.w_arを求めるためのZ行列
    # date_ar   : z_ar0の要素 
    #------------------------------------------------------------
    def AR(self,y,k):
        start = time.time()
        date_ar = []
        z_ar1 = []
        for i in range(self.p):
            z_ar0 = []
            for j in range(self.N-self.d):
                date_ar = np.array((self.kDate['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                z_ar0.append(float(self.kDate[self.kDate['date'] == date_ar[-1]]['hll']))
            z_ar1.append(z_ar0)
        z_ar1 = np.array(z_ar1).T
        z_ar1 = np.append(z_ar1, np.ones([z_ar1.shape[0],1]),axis=1)
        
        sigma_ar0 = np.matmul(z_ar1.T, z_ar1)
        sigma_ar1 = np.matmul(z_ar1.T, y) 
        w_ar_buf = np.matmul(np.linalg.inv(sigma_ar0), sigma_ar1)
        self.w_ar = np.append(self.w_ar, w_ar_buf).reshape([self.p+1,k+1])
        end_time = time.time() - start
        print("time_AR : {0}".format(end_time) + "[sec]")  
        print('w_ar :', k)
        print(self.w_ar)
    
    def MA(self,e,k):
        start = time.time()
        z_ma1 = []
        for i in range(self.q):
            z_ma0 = []
            for j in range(self.N-self.d):
                z_ma0.append(self.eps[(j+i+2):(j+i+3)][0])
            z_ma1.append(z_ma0)
        z_ma1 = np.array(z_ma1).T
        z_ma1 = np.append(z_ma1, np.ones([z_ma1.shape[0],1]),axis=1)

        sigma_ma0 = np.matmul(z_ma1.T, z_ma1)
        sigma_ma1 = np.matmul(z_ma1.T, e)
        w_ma_buf = np.matmul(np.linalg.inv(sigma_ma0), sigma_ma1)
        self.w_ma = np.append(self.w_ma, w_ma_buf).reshape([self.q+1, k+1])
        end_time = time.time() - start
        print("time_MA : {0}".format(end_time) + "[sec]") 
        print('w_ma :', k)
        print(self.w_ma)

    #------------------------------------------------------------
    # ARIMAモデルの学習
    # date_ar :
    # selfi.kDate['hll'][self.k.index[0]] 
    #------------------------------------------------------------
    def train(self):
        start_all = time.time()
        for k in range(int(self.tData['krage'][-1:]) - 10000):
            self.kDate = self.tData[self.tData['krage']==10000+k]
            #self.k = self.kDate[self.kDate['date'] == '2018-03-31']
            
            y = []
            date_y = None
            e = []
            for i in range(self.N-self.d):
                date_y = np.array((self.kDate['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
                if i == 0:
                    y.append(float(self.kDate[self.kDate['date'] == date_y[-1]]['hll']))
                    e.append(self.eps[i:(i+1)][0])
                else:
                    y.append(float(self.kDate[self.kDate['date'] == date_y[-1]]['hll']))
                    y[i-1] = y[i-1] - y[i]
                    e.append(self.eps[i:(i+1)][0])
                    e[i-1] = e[i-1] - e[i]
            y = np.array(y)[np.newaxis].T
            e = np.array(e)[np.newaxis].T
            
            #z_ar0.append(float(self.kDate[self.kDate['date'] == date_ar[-1]]['hll']))
            #with concurrent.futures.ProcessPoolExecutor() as executor:
            #    executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
            #    executor.submit(self.AR,y,k)
            #    executor.submit(self.MA,e,k)
            self.AR(y,k)
            self.MA(e,k)

        end_time = time.time() - start_all
        print("time : {0}".format(end_time) + "[sec]") 

    def multi_train(self):
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as executor:
            executor.submit(self.train)

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
    ar_w_list = []
    ma_w_list = []

    for no in range(len(fileind)):
        arima = Arima(mytrackData.train_xData[0],mytrackData.train_tData[0])
        # ar_list.append(ar)
        arima.train()
        arima.w_ar.tolist()
        arima.w_ma.tolist()
        pdb.set_trace()
        ar_w_list.append(arima.w_ar)
        ma_w_list.append(arima.w_ma)

    f_ar = open("ar_w_list.binaryfile","wb")
    f_ma = open("ar_w_list.binaryfile","wb")
    pickle.dump(ar_w_list,f)
    pickle.dump(ma_w_list,f)
    f.close
