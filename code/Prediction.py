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
#self.days : 入力データの総日数
#self.krage_length : 入力データの総キロ数
#
#------------------------------------
class prediction():
    def __init__(self,w,x,t):
        self.p = 10
        self.N = 50
        self.x = x
        self.t = t

        sdate = t.head(1)['date'].iat[0]
        edate = t.tail(1)['date'].iat[0]

        self.xNum = self.t.shape[0]
        self.days = (edate - sdate + dt.timedelta(days=1)).days
        self.krage_length = int(self.xNum/self.days) #キロ程の総数
        self.w = w
        # self.w = np.random.normal(0.0, pow(100, -0.5), (self.p + 1, 1)) #動作確認用のランダムなｗ


    def predict(self,day):
        #pdb.set_trace()
        aDay = dt.timedelta(days=1)
        y = []
        tmp = []
        for i in range(self.p):
            date = day - aDay * (i+1)
            y = np.append(y,self.t[self.t['date'] == date]['hll'])

        y = y.reshape([self.p,self.krage_length])

        y = self.w[0] + np.matmul(self.w[1:].T, y)

        df = pd.DataFrame(y.T,columns=['hll'])

        #'date'をdfの末尾に追加
        df['date'] = day

        df = df.ix[:,['date','hll']] #'date','hll'の順番に並び替え

        self.t = pd.concat([self.t,df])
        return self.t

    def loss(self,tDate):
        t = np.array(tDate['hll'])[np.newaxis]
        pdb.set_trace()
        #t = t[t['date'] == '2018-03-31']
        num = pow(t - self.predict(tDate),2)
        loss = np.sum(num) / (t.shape[1])
        return loss

    def showY(self,x,y):
        plt.plot(x,y,'-o',color="0000FF")
        plt.xlabel("date")
        plt.ylabel("hll")
        plt.legend(["hll"])
        plt.show()

class trackData():
    def __init__(self):
        self.xTrain_list = []
        self.tTrain_list = []
        self.fileind = ['A','B','C','D']
        self.fNum = len(self.fileind)
        for no in range(self.fNum):
            self.load_file("w_list.binaryfile",self.w_list)
            fname_xTra = "xTrain_{}.binaryfile".format(self.fileind[no])
            fname_tTra = "tTrain_{}.binaryfile".format(self.fileind[no])
            self.load_file(fname_xTra,self.xTrain_list)
            self.load_file(fname_tTra,self.tTrain_list)

    def load_file(self,filename,data):
        f = open(filename,'rb')
        data.append(pickle.load(f))
        f.close


if __name__ == "__main__":

    myData=trackData() #Trainとwのリストを読み込む

    aDay = dt.timedelta(days=1)
    sDate = dt.datetime(2018,4,1,00,00,00)
    eDate = dt.datetime(2018,6,30,00,00,00)

    nite = (eDate-sDate + aDay).days #予測する日数(int)

    fNum = myData.fNum #ファイルの数（A~Dの４つ）
    y = [] #予測した高低左(A~Dの４つ)を格納

    for j in range(fNum):
        pre = prediction(myData.w_list[j],myData.xTrain_list[j],myData.tTrain_list[j])
        # pre = prediction(0,myData.xTrain_list[j],myData.tTrain_list[j]) #動作確認用

        for i in range(nite):
            date = sDate + i*aDay
            pre.predict(date)

        out = pre.t.iloc[pre.xNum:]
        y.append(out)
    # pre.showY(range(nite),y)

    for i in range(myData.fNum):
        fname = "output_{}.csv".format(myData.fileind[i])
        f = open(fname,"w")
        y[i].to_csv(fname)
        f.close
