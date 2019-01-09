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
#   https://logics-of-blue.com/var%E3%83%A2%E3%83%87%E3%83%AB/ を参照
#
# self.N     : 使用する日数
# self.p     : 遡る日数(ARモデル)
# self.q     : 遡る日数(MAモデル)
# self.d     : d次階差時系列
# self.w_l_var  : VARモデルでの目的変数(hll)のパラメータ(重み)
# self.w_r_var  : VARモデルでの説明変数(hlr)のパラメータ(重み)
# self.kData : キロ程ごとのデータを格納する変数
#
# amount     : 扱うデータの総日数(365日とか)
class Arima():
    def __init__(self,xData,tData):
        self.xData = xData
        self.tData = tData

        self.xDim = xData.shape[1]-1
        self.xNum = xData.shape[0]

        self.tNum = tData.shape[0]

        self.right = xData[0]

        self.kData = []
        self.N = 10
        self.p = 10
        self.s = -(91+self.p)

        self.krage_length = self.tData.shape[1]
        self.w_l_var = []
        self.w_r_var = []

    #------------------------------------------------------------
    # VARモデル :
    #   時系列データの目的変数と説明変数(hlrのみ)を使用して将来の予測を行う回帰
    #   self.N日のデータからself.p日遡って回帰を行う
    #
    # z_l_var0     : self.w_l_varを求めるためのZ行列の列の要素
    # z_r_var0     : self.w_r_varを求めるためのZ行列の列の要素
    # z_l_var1     : self.w_l_varを求めるためのZ行列
    # z_r_var1     : self.w_r_varを求めるためのZ行列
    #
    # self.w_l_var : (Z^T * Z + λI)^-1 * Z^T * y
    # self.w_r_var : (Z^T * Z + λI)^-1 * Z^T * y
    def VAR(self,y_l,y_r,k):
        start = time.time()
        z_l_var1 = []
        z_r_var1 = []
        
        #--------------------------------------------------
        # VARモデルのパラメータを更新するための行列Zを導出
        # https://logics-of-blue.com/var%E3%83%A2%E3%83%87%E3%83%AB/ を参照
        # for文の段階では p x (N-d) 行列
        for i in range(self.p):
            z_l_var0 = []
            z_r_var0 = []
            for j in range(self.N):
                # z_l_var0にj+i+2日前のデータを格納
                z_l_var0.append(float(self.kData[j+i+self.s-2]))
                z_r_var0.append(float(self.rData[j+i+self.s-2]))
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

        sigma_l_var1 = np.matmul(z_l_var1.T, y_l)
        sigma_r_var1 = np.matmul(z_r_var1.T, y_r)
        
        w_l_var_buf = np.matmul(np.linalg.inv(sigma_l_var0), sigma_l_var1)
        w_r_var_buf = np.matmul(np.linalg.inv(sigma_r_var0), sigma_r_var1)

        if k == 0:
            self.w_l_var = np.append(self.w_l_var, w_l_var_buf)[np.newaxis].T 
            self.w_r_var = np.append(self.w_r_var, w_r_var_buf)[np.newaxis].T 
        else:
            self.w_l_var = np.append(self.w_l_var, w_l_var_buf, axis=1)
            self.w_r_var = np.append(self.w_r_var, w_r_var_buf, axis=1)
        #-----------------------------------------------------------
        
        end_time = time.time() - start
        print("time_AR : {0}".format(end_time) + "[sec]")
        print('w_r_var :', k)
        print(self.w_r_var)
        print('w_l_var :', k)
        print(self.w_l_var)
    #------------------------------------------------------------

    #------------------------------------------------------------
    # VARモデルの学習
    # 
    # y : 1 ~ N 日前の時系列データを格納した行列(ベクトル)
    def train(self):
        start_train = time.time()
        #------------------------------------------------------------
        # 各キロ程ごとの時系列データ(amount日分)を取得し、y・e 行列(ベクトル)を作成
        # 行列の掛け算を行うために[np.newaxis]をy・e行列(ベクトル)にかけている
        for k in range(self.krage_length):
            self.kData = self.tData[:,k]
            self.rData = self.right[:,k]

            y_l = []
            y_r = []
            for i in range(self.N):
                y_l.append(float(self.kData[i+self.s-1]))
                y_r.append(float(self.kData[i+self.s-1]))
            y_l = np.array(y_l)[np.newaxis].T
            y_r = np.array(y_r)[np.newaxis].T

            #------------------------------------------------------------
            # VARモデルの計算
            self.VAR(y_l,y_r,k)
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

    w_l_var_list = []
    w_r_var_list = []

    start_all = time.time()
    for no in range(len(fileind)):
        # Arimaクラスのインスタンス化
        arima = Arima(mytrackData.train_xData[no],mytrackData.train_tData[no])
        # Arimaモデルの学習
        arima.train()
        #--------------------------------------
        # VARで求めた重みを出力するためにリストに格納
        w_l_var_list.append(arima.w_l_var)
        w_r_var_list.append(arima.w_r_var)
        #--------------------------------------
    end_time = time.time() - start_all
    print("time : {0}".format(end_time) + "[sec]")
    
    #--------------------------------------
    # VARで求めた重みのリストをファイル出力
    f_l_var = open("w_l_var_list.binaryfile","wb")
    f_r_var = open("w_r_var_list.binaryfile","wb")
    pickle.dump(w_l_var_list,f_l_var)
    pickle.dump(w_r_var_list,f_r_var)
    f_l_var.close()
    f_r_var.close()
    #--------------------------------------
