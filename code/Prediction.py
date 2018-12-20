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
        self.p = 50
        self.N = 50
        self.x = x
        self.t = t

        sDate = t.head(1)['date']
        eDate = t.tail(1)['date']

        self.days = sDate - eDate + dt.timedelta(days=1)
        self.krage_length = int(self.t.shape[0]/self.days) #キロ程の総数
        # self.w = w
        self.w = np.random.normal(0.0, pow(100, -0.5), (self.p + 1, 1)) #動作確認用のランダムなｗ


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

        df = pd.DataFrame(y,columns=['hll'])

        #'krage','date'をdfの末尾に追加
        df['krage'] = t['krage'].value
        df['date'] = day

        df.ix[:,['date','krage','hll']] #'date','krage','hll'の順番に並び替え

        self.t = pd.concat(self.t,df)
        return y

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
        fileind = ['A','B','C','D']
        self.fNum = len(fileind)
        for no in range(self.fNum):
            #self.load_file("w_list.binaryfile",self.w_list)
            fname_xTra = "xTrain_{}.binaryfile".format(fileind[no])
            fname_tTra = "tTrain_{}.binaryfile".format(fileind[no])
            self.load_file(fname_xTra,self.xTrain_list)
            self.load_file(fname_tTra,self.tTrain_list)

    def load_file(self,filename,data):
        f = open(filename,'rb')
        data.append(pickle.load(f))
        f.close


if __name__ == "__main__":

    myData=trackData() #Trainとwのリストを読み込む

    aDay = dt.timedelta(days=1)
    sDate = dt.date(2018,4,1)
    eDate = dt.date(2018,6,30)

    nite = (eDate-sData).days #予測する日数

    fNum = myData.fNum #ファイルの数（A~Dの４つ）
    y = [] #予測した高低左を格納

    for j in range(fNum):
        # pre = prediction(myData.w_list[j],myData.xTrain_list[j],myData.tTrain_list[j],days[j])
        pre = prediction(0,myData.xTrain_list[j],myData.tTrain_list[j],days[j]) #動作確認用

        for i in range(nite):
            date = sDate + i*aDay
            y.append(pre.predict(date))

    # pre.showY(range(nite),y)

    output = pd.DataFrame([y],columns="高低左")
    f = open("output.csv","w")
    pickle.dump(f,output)
    f.close
