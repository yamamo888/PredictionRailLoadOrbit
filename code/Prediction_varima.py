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
    def __init__(self,w_l_var,w_r_var,w_l_vma,w_r_vma,x,t,eps_l,eps_r):
        self.p = 10
        self.q = 10
        self.N = 10
        self.s = 91
        self.x = x[0] #xTrainデータ:hlr
        self.t = t #tTrainデータ:hll
        self.days = self.t.shape[0] #日数
        self.krage_length = self.t.shape[1] #キロ程の総数
        self.w_l_var = w_l_var
        self.w_r_var = w_r_var
        self.w_l_vma = w_l_vma
        self.w_r_vma = w_r_vma
        self.eps_l = eps_l
        self.eps_r = eps_r
        # self.w = np.random.normal(0.0, pow(100, -0.5), (self.p + 1, 1)) #動作確認用のランダムなｗ

    def predict(self):
        y_l = self.t[-(self.p + self.s):-self.s]
        y_r = self.x[-(self.p + self.s):-self.s]
        y_var = self.w_l_var[0] + self.w_r_var[0] + np.sum(self.w_l_var[1:]*y_l,axis=0) + np.sum(self.w_r_var[1:]*y_r,axis=0) 
        y_vma = -np.sum(self.w_l_vma[1:]*self.eps_l[1:self.q+1],axis=0) - np.sum(self.w_r_vma[1:]*self.eps_r[1:self.q+1],axis=0) + self.eps_l[0] + self.eps_r[0]
        y =  y_var + y_vma
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
        self.w_l_var_list = self.load_file("w_l_var_list.binaryfile")
        self.w_r_var_list = self.load_file("w_r_var_list.binaryfile")

        self.w_l_vma_list = self.load_file("w_l_vma_list.binaryfile")
        self.w_r_vma_list = self.load_file("w_r_vma_list.binaryfile")

        self.eps_l_list = self.load_file("eps_l_list.binaryfile")
        self.eps_r_list = self.load_file("eps_r_list.binaryfile")

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
        pre = prediction(myData.w_l_var_list[j],myData.w_r_var_list[j],myData.w_l_vma_list[j],myData.w_r_vma_list[j],myData.xTrain_list[j],myData.tTrain_list[j],myData.eps_l_list[j],myData.eps_r_list[j])
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

    output.index = range(output.shape[0])
    f = open("output.csv","w")
    output.to_csv(f)
    f.close()