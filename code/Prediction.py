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

#------------------------------------
#self.x :
#self.t : 高低左
#self.w : 予測パラメータ
#self.p : 予測のためにさかのぼる日数
#self.q :
#self.days : 入力データの総日数
#self.krage_length : 入力データの総キロ数
#self.eps : 学習で使用したホワイトノイズ
#------------------------------------
class prediction():
    def __init__(self,w_ar,w_ma,x,t,eps):
        self.p = 10
        self.q = 10
        self.N = 10
        self.s = 91
        self.x = x #xTrainデータ:hll以外
        self.t = t #tTrainデータ:hll
        self.days = self.t.shape[0] #日数
        self.krage_length = self.t.shape[1] #キロ程の総数
        self.w_ar = w_ar
        self.w_ma = w_ma
        self.eps = eps
        # self.w = np.random.normal(0.0, pow(100, -0.5), (self.p + 1, 1)) #動作確認用のランダムなｗ

    def predict(self):
        y = self.t[-(self.p + self.s):-self.s]
        y = self.w_ar[0] + np.sum(self.w_ar[1:]*y,axis=0) - np.sum(self.w_ma*self.eps[1:self.q+2],axis=0) + self.eps[0]
        #y = self.w_ar[0] + np.sum(self.w_ar[1:]*y,axis=0) + self.eps[0]
        #pdb.set_trace()
        y = y.reshape(1,y.shape[0])
        self.t = np.append(self.t,y,axis=0)

    # def loss(self,tDate):
    #     t = np.array(tDate['hll'])[np.newaxis]
    #     pdb.set_trace()
    #     #t = t[t['date'] == '2018-03-31']
    #     num = pow(t - self.predict(tDate),2)
    #     loss = np.sum(num) / (t.shape[1])
    #     return loss

    def showY(self,x,y):
        plt.plot(x,y,'-o',color="0000FF")
        plt.xlabel("date")
        plt.ylabel("hll")
        plt.legend(["hll"])
        plt.show()

class trackData():
    def __init__(self):
        self.ar_w_list = self.load_file("ar_w_list.binaryfile")
        self.ma_w_list = self.load_file("ma_w_list.binaryfile")
        self.eps_list = self.load_file("eps_list.binaryfile")
        self.xTrain_list = []
        self.tTrain_list = []
        self.fileind = ['A','B','C','D']
        self.fNum = len(self.fileind)

        for no in range(self.fNum):
            fname_xTra = "../data/track_xTrain_{}.binaryfile".format(self.fileind[no])
            fname_tTra = "../data/track_tTrain_{}.binaryfile".format(self.fileind[no])
            self.xTrain_list.append(self.load_file(fname_xTra))
            self.tTrain_list.append(self.load_file(fname_tTra))

    def load_file(self,filename):
        f = open(filename,'rb')
        result = pickle.load(f)
        f.close
        return result


if __name__ == "__main__":

    myData=trackData() #Trainとwのリストを読み込む
    nite = 91 #予測する日数
    fNum = myData.fNum #ファイルの数（A~Dの４つ）
    y = [] #予測した高低左(A~Dの４つ)を格納

    for j in range(fNum):
        pre = prediction(myData.ar_w_list[j],myData.ma_w_list[j],myData.xTrain_list[j],myData.tTrain_list[j],myData.eps_list[j])
        # pre = prediction(0,myData.xTrain_list[j],myData.tTrain_list[j]) #動作確認用
        for _ in range(nite):
            pre.predict() #次の日を予測する
        out = pre.t[pre.days:]
        out = pd.DataFrame(out.reshape(out.shape[0]*out.shape[1],1))
        #pdb.set_trace()
        #y.append(out[:,1])
        y.append(out[:])

    output = y[0]
    pdb.set_trace()
    for i in range(1,fNum):
        output = pd.concat([output,y[i]],axis = 0)

    f = open("output.csv","w")
    output.to_csv(f)
    f.close()
