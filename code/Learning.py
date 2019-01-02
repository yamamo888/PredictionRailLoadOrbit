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
# self.w_ar  : ARモデルでのパラメータ(重み)
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

        #self.t = self.tData[self.tData['date'] == '2018-3-31']
        self.kData = []
        self.kEps = []
        self.N = 10
        self.p = 3
        self.q = 3
        self.d = 1

        #self.krage_length = xData[xData["date"] == dt.datetime(2017,4,10,00,00,00)]["krage"].shape[0]
        self.krage_length = self.tData.shape[1]
        self.w_ar = []
        self.w_ma = []

        self.eps = np.random.normal(1,25,(amount,self.krage_length))

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
    def AR(self,y,k):
        start = time.time()
        date_ar = []
        z_ar1 = []
        
        #--------------------------------------------------
        # ARモデルのパラメータを更新するための行列Zを導出
        # htitps://k-san.link/ar-process/を参照
        # for文の段階では p x (N-d) 行列
        for i in range(self.p):
            z_ar0 = []
            for j in range(self.N-self.d):
                # date_arにj+i+2日前の日にち(str型)を保存
                #date_ar = np.array((self.kData['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                # date_arに保存されている日にちに対応するデータを列方向に追加
                #z_ar0.append(float(self.kData[self.kData['date'] == date_ar[-1]]['hll']))
                z_ar0.append(float(self.kData[j+i+2]))
            # 行方向にz_ar0を追加しZ行列を生成
            z_ar1.append(z_ar0)        
        # p x (N-d) -> (N-d) x p (Z行列はN x p)
        z_ar1 = np.array(z_ar1).T
        # y = wx + b の線形回帰の式のbをWに融合させるために最後の列に1の列を追加
        z_ar1 = np.append(z_ar1, np.ones([z_ar1.shape[0],1]),axis=1)
        #--------------------------------------------------


        #-----------------------------------------------------------
        # self.w_arの更新
        sigma_ar0 = np.matmul(z_ar1.T, z_ar1)
        # 逆行列を生成するためにλIを足す(対角成分にλを足す)
        sigma_ar0 += 0.0000001 * np.eye(sigma_ar0.shape[0])
        sigma_ar1 = np.matmul(z_ar1.T, y)
        w_ar_buf = np.matmul(np.linalg.inv(sigma_ar0), sigma_ar1)
        self.w_ar = np.append(self.w_ar, w_ar_buf).reshape([self.p+1,k+1]) 
        #-----------------------------------------------------------
       
        
        end_time = time.time() - start
        print("time_AR : {0}".format(end_time) + "[sec]")
        print('w_ar :', k)
        print(self.w_ar)
    #------------------------------------------------------------

    #------------------------------------------------------------
    # MAモデル :
    #   ホワイトノイズ(ε)で将来の予測を行う回帰
    #   self.N日のデータからself.p日遡って回帰を行う
    #
    # z_ma0     : self.w_maを求めるためのZ行列の列の要素
    # z_ma1     : self.w_maを求めるためのZ行列
    # date_ma   : z_ma0の１要素
    #
    # self.w_ma : (Z^T * Z + λI)^-1 * Z^T * y
    def MA(self,e,k):
        start = time.time()
        z_ma1 = []
        
        #-------------------------------------------------------------
        # MAモデルのパラメータを更新するための行列Zを導出
        # htitps://k-san.link/ar-process/を参照(ARモデルと基本同じ(多分))
        # このfor文の段階では q x (N-d) 行列
        for i in range(self.q):
            z_ma0 = []
            for j in range(self.N-self.d):
                z_ma0.append(self.kEps[(j+i+2):(j+i+3)][0])
            z_ma1.append(z_ma0)

        # p x (N-d) -> (N-d) x p (Z行列は(N-d) x p)
        z_ma1 = np.array(z_ma1).T
        # y = wx + b の線形回帰の式のbをWに融合させるために最後の列に1の列を追加
        z_ma1 = np.append(z_ma1, np.ones([z_ma1.shape[0],1]),axis=1)
        #-------------------------------------------------------------

        #-----------------------------------------------------------
        # self.w_maの更新
        sigma_ma0 = np.matmul(z_ma1.T, z_ma1)
        sigma_ma0 += 0.0000001 * np.eye(sigma_ma0.shape[0])
        sigma_ma1 = np.matmul(z_ma1.T, e)
        w_ma_buf = np.matmul(np.linalg.inv(sigma_ma0), sigma_ma1)
        self.w_ma = np.append(self.w_ma, w_ma_buf).reshape([self.q+1, k+1])
        #-----------------------------------------------------------

        end_time = time.time() - start
        print("time_MA : {0}".format(end_time) + "[sec]")
        print('w_ma :', k)
        print(self.w_ma)
    #------------------------------------------------------------

    #------------------------------------------------------------
    # ARIMAモデルの学習
    # 
    # y : 1 ~ N 日前の時系列データを格納した行列(ベクトル)
    # e : 1 ~ N 日前のホワイトノイズを格納した行列(ベクトル)
    def train(self):
        start_train = time.time()
        #------------------------------------------------------------
        # 各キロ程ごとの時系列データ(amount日分)を取得し、y・e 行列(ベクトル)を作成
        # 行列の掛け算を行うために[np.newaxis]をy・e行列(ベクトル)にかけている
        for k in range(self.krage_length):
            #self.kData = self.tData[self.tData['krage']==10000+k]
            #pdb.set_trace()
            #self.kData = self.tData[k]
            #self.kEps = self.eps[:,k]
            #self.k = self.kData[self.kData['date'] == '2018-03-31']
            self.kData = self.tData[:,k]
            self.kEps = self.eps[:,k]
            #date_y = None

            y = []
            e = []
            for i in range(self.N):
                #date_y = np.array((self.kData['date'][-1:] - datetime.timedelta(days=i+1)).astype(str))
                #------------------------------------------------------------
                # 1階差分の計算
                # 1番目の要素は普通に計算、それ以降は一つ前の要素から引いたものをリストに格納
                if i == 0:
                    #y.append(float(self.kData[self.kData['date'] == date_y[-1]]['hll']))
                    #pdb.set_trace()
                    y.append(float(self.kData[i+1]))
                    e.append(self.kEps[i])
                else:
                    #y.append(float(self.kData[self.kData['date'] == date_y[-1]]['hll']))
                    y.append(float(self.kData[i+1]))
                    y[i-1] = y[i-1] - y[i]
                    e.append(self.kEps[i])
                    e[i-1] = e[i-1] - e[i]
                #------------------------------------------------------------
            y = np.array(y[:self.N-self.d])[np.newaxis].T
            e = np.array(e[:self.N-self.d])[np.newaxis].T

            #pdb.set_trace()
            #------------------------------------------------------------
            # ARモデルとMAモデルの計算
            self.AR(y,k)
            self.MA(e,k)
            #------------------------------------------------------------

        #------------------------------------------------------------

        end_time = time.time() - start_train
        print("time : {0}".format(end_time) + "[sec]")
    #------------------------------------------------------------

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

    # ar_list = []
    ar_w_list = []
    ma_w_list = []
    eps_list = []

    start_all = time.time()
    #pdb.set_trace()
    for no in range(len(fileind)):
        #pdb.set_trace()
        arima = Arima(mytrackData.train_xData[no],mytrackData.train_tData[no])
        # ar_list.append(ar)
        arima.train()
        #arima.w_ar = arima.w_ar.tolist()
        #arima.w_ma = arima.w_ma.tolist()
        #print(type(arima.w_ar))
        #pdb.set_trace()
        ar_w_list.append(arima.w_ar)
        ma_w_list.append(arima.w_ma)
        eps_list.append(arima.eps)
    end_time = time.time() - start_all
    print("time : {0}".format(end_time) + "[sec]")

    f_ar = open("ar_w_list.binaryfile","wb")
    f_ma = open("ma_w_list.binaryfile","wb")
    f_eps = open("eps_list.binaryfile","wb")
    pickle.dump(ar_w_list,f_ar)
    pickle.dump(ma_w_list,f_ma)
    pickle.dump(eps_list,f_eps)
    f_ar.close()
    f_ma.close()
    f_eps.close()
