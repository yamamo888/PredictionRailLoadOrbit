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
#   ARモデルとMAモデルを融合させ、d階差時系列を採用したもの
#   htitps://k-san.link/ar-process/を参照
#
# self.N     : 使用する日数
# self.p     : 遡る日数(ARモデル)
# self.q     : 遡る日数(MAモデル)
# self.d     : d次階差時系列
# self.w_l_var  : ARモデルでのパラメータ(重み)
# self.w_r_var  : ARモデルでのパラメータ(重み)
# self.w_ma  : MAモデルでのパラメータ(重み)
# self.eps   : MAモデルで使用するホワイトノイズ
# self.kData : キロ程ごとのデータを格納する変数
# self.kEps  : キロ程ごとの誤差項を格納する変数
#
# ns_to_day  : [ns]をdayに変換するための変数
# amount     : 扱うデータの総日数(365日とか)
class Arima():
    def __init__(self,xData,tData):
        self.xData = xData
        self.tData = tData

        self.xDim = xData.shape[1]-1
        self.xNum = xData.shape[0]

        self.tNum = tData.shape[0]

        # 秒を日にちに変換(86400 -> 1 みたいに)
        #ns_to_day = 86400000000000
        #amount = int((self.tData['date'][-1:].values - self.tData['date'][0:1].values)[0]/ns_to_day)+1
        #pdb.set_trace()
        amount = self.tData.shape[0]

        self.rData = xData[0]

        #self.t = self.tData[self.tData['date'] == '2018-3-31']
        self.kData = []
        self.N = 10
        self.p = 10
        self.s = 91

        #self.krage_length = xData[xData["date"] == dt.datetime(2017,4,10,00,00,00)]["krage"].shape[0]
        self.krage_length = self.tData.shape[1]
        self.w_l_var = []

        self.eps = np.random.normal(1,4,(amount,self.krage_length))

    #------------------------------------------------------------
    # ARモデル :
    #   時系列データの目的変数のみで将来の予測を行う回帰
    #   self.N日のデータからself.p日遡って回帰を行う
    #
    # z_ar0     : self.w_arを求めるためのZ行列の列の要素
    # z_ar1     : self.w_arを求めるためのZ行列
    # date_ar   : z_ar0の１要素
    #
    # self.w_ar : (Z^T * Z + λI)^-1 * Z^T * y
    def VAR(self,y,k):
        start = time.time()
        z_l_var1 = []
        z_r_var1 = []
        
        #--------------------------------------------------
        # ARモデルのパラメータを更新するための行列Zを導出
        # htitps://k-san.link/ar-process/を参照
        # for文の段階では p x (N-d) 行列
        for i in range(self.p):
            z_l_var0 = []
            z_r_var0 = []
            for j in range(self.N):
                # z_l_var0にj+i+2日前のデータを格納
                z_l_var0.append(float(self.kData[j+i+2+self.s]))
                z_r_var0.append(float(self.rData[j+i+2+self.s]))
            # 行方向にz_ar0を追加しZ行列を生成
            z_l_var1.append(z_l_var0)
            z_r_var1.append(z_r_var0)
        # p x (N-d) -> (N-d) x p (Z行列はN x p)
        z_l_var1 = np.array(z_l_var1).T
        z_r_var1 = np.array(z_r_var1).T
        # y = wx + b の線形回帰の式のbをWに融合させるために最後の列に1の列を追加
        z_l_var1 = np.append(z_l_var1, np.ones([z_l_var1.shape[0],1]),axis=1)
        z_r_var1 = np.append(z_r_var1, np.ones([z_r_var1.shape[0],1]),axis=1)
        #--------------------------------------------------

        #-----------------------------------------------------------
        # self.w_arの更新
        sigma_l_var0 = np.matmul(z_l_var1.T, z_l_var1)
        sigma_r_var0 = np.matmul(z_r_var1.T, z_r_var1)
        # 逆行列を生成するためにλIを足す(対角成分にλを足す)
        sigma_l_var0 += 0.0000001 * np.eye(sigma_l_var0.shape[0])
        sigma_r_var0 += 0.0000001 * np.eye(sigma_r_var0.shape[0])

        sigma_l_var1 = np.matmul(z_l_var1.T, y)
        sigma_r_var1 = np.matmul(z_r_var1.T, y)
        
        w_l_var_buf = np.matmul(np.linalg.inv(sigma_l_var0), sigma_l_var1)
        w_r_var_buf = np.matmul(np.linalg.inv(sigma_r_var0), sigma_r_var1)
        #self.w_ar = np.append(self.w_ar, w_ar_buf).reshape([self.p+1,k+1]) 
        if k == 0:
            self.w_l_var = np.append(self.w_l_var, w_l_var_buf)[np.newaxis].T 
            self.w_r_var = np.append(self.w_r_var, w_r_var_buf)[np.newaxis].T 
        else:
            self.w_l_var = np.append(self.w_l_var, w_l_var_buf, axis=1)
            self.w_r_var = np.append(self.w_r_var, w_r_var_buf, axis=1)
        #-----------------------------------------------------------
        
        end_time = time.time() - start
        print("time_AR : {0}".format(end_time) + "[sec]")
        print('w_l_var :', k)
        print(self.w_ar)
    #------------------------------------------------------------

    #------------------------------------------------------------
    # ARIMAモデルの学習
    # 
    # y : 1 ~ N 日前の時系列データを格納した行列(ベクトル)
    def train(self):
        start_train = time.time()
        #------------------------------------------------------------
        # 各キロ程ごとの時系列データ(amount日分)を取得し、y・e 行列(ベクトル)を作成
        # 行列の掛け算を行うために[np.newaxis]をy・e行列(ベクトル)にかけている
        for k in range(self.krage_length):
            #self.kData = self.tData[self.tData['krage']==10000+k]
            #self.kData = self.tData[k]
            #self.kEps = self.eps[:,k]
            #self.k = self.kData[self.kData['date'] == '2018-03-31']
            self.kData = self.tData[:,k]

            y = []
            for i in range(self.N):
                #date_y = np.array((self.kData['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
                y.append(float(self.kData[i+1+self.s]))
            y = np.array(y)[np.newaxis].T

            #------------------------------------------------------------
            # ARモデルとMAモデルの計算
            self.VAR(y,k)
            #------------------------------------------------------------

        #------------------------------------------------------------

        end_time = time.time() - start_train
        print("time : {0}".format(end_time) + "[sec]")
    #------------------------------------------------------------

#------------------------------------------------------------

class trackData():
    dataPath = '../data'
    def __init__(self):#trainの読み込み

        self.train_xData = []
        # self.test_xData = []
        self.train_tData = []
        # self.test_tData = []

        fileind = ['A','B','C','D']

        for no in range(len(fileind)):
            fname_xTra = "track_xTrain_{}.binaryfile".format(fileind[no])
            # fname_xTes = "xTest_{}.binaryfile".format(fileind[no])
            fname_tTra = "track_tTrain_{}.binaryfile".format(fileind[no])
            # fname_tTes = "tTest_{}.binaryfile".format(fileind[no])

            self.load_file(fname_xTra,self.train_xData)
            # self.load_file(fname_xTes,self.test_xData)
            self.load_file(fname_tTra,self.train_tData)
            # self.load_file(fname_tTes,self.test_tData)


    def load_file(self,filename,data):
        fullpath = os.path.join(self.dataPath, filename)
        f = open(fullpath,'rb')
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

    ar_w_list = []

    start_all = time.time()
    for no in range(len(fileind)):
        arima = Arima(mytrackData.train_xData[no],mytrackData.train_tData[no])
        arima.train()
        ar_w_list.append(arima.w_ar)
    end_time = time.time() - start_all
    print("time : {0}".format(end_time) + "[sec]")

    f_ar = open("ar_w_list.binaryfile","wb")
    pickle.dump(ar_w_list,f_ar)
    f_ar.close()
