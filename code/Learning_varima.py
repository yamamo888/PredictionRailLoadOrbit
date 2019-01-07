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
# VARIMAモデルでの自己回帰 :
#   VARモデルとVMAモデルを融合させ、d階差時系列を採用したもの
#   https://logics-of-blue.com/var%E3%83%A2%E3%83%87%E3%83%AB/ を参照
#
# self.N        : 使用する日数
# self.p        : 遡る日数(ARモデル)
# self.q        : 遡る日数(MAモデル)
# self.d        : d次階差時系列
# self.w_l_var  : VARモデルでの目的変数(hll)のパラメータ(重み)
# self.w_r_var  : VARモデルでの説明変数(hlr)のパラメータ(重み)
# self.w_l_vma  : VMAモデルでの目的変数(hll)のパラメータ(重み)
# self.w_r_vma  : VMAモデルでの説明変数(hlr)のパラメータ(重み)
# self.eps_l    : VMAモデルで使用するホワイトノイズ(目的変数(hll))
# self.eps_r    : VMAモデルで使用するホワイトノイズ(説明変数(hlr))
# self.kData    : キロ程ごとのデータを格納する変数
# self.kEps     : キロ程ごとの誤差項を格納する変数
#
# amount        : 扱うデータの総日数(365日とか)
class Varima():
    def __init__(self,xData,tData):
        self.tData = tData

        self.xDim = xData.shape[1]-1
        self.xNum = xData.shape[0]

        self.tNum = tData.shape[0]

        amount = self.tData.shape[0]

        self.explain = []
        for i in range(len(xData)):
            self.explain.append(xData[i])

        self.kData = []
        self.k_tEps = []
        self.k_xEps = []
        self.N = 10
        self.p = 10
        self.q = 10
        self.d = 1
        self.s = 91

        self.krage_length = self.tData.shape[1]
        
        self.w_l_var = []
        self.w_r_var = []

        self.w_l_vma = []
        self.w_r_vma = []
        
        self.eps_r = np.random.normal(1,4,(amount,self.krage_length))
        self.eps_l = np.random.normal(1,4,(amount,self.krage_length))

    #------------------------------------------------------------
    # VARモデル :
    #   時系列データの目的変数のみで将来の予測を行う回帰
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
            for j in range(self.N-self.d):
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
    # VMAモデル :
    #   目的変数(hll)と説明変数(hlr)のホワイトノイズ(ε)で将来の予測を行う回帰
    #   self.N日のデータからself.p日遡って回帰を行う
    #
    # z_l_vma0     : self.w_l_vmaを求めるためのZ行列の列の要素
    # z_r_vma0     : self.w_r_vmaを求めるためのZ行列の列の要素
    # z_l_vma1     : self.w_l_vmaを求めるためのZ行列
    # z_r_vma1     : self.w_r_vmaを求めるためのZ行列
    #
    # self.w_l_vma : (Z^T * Z + λI)^-1 * Z^T * y
    # self.w_r_vma : (Z^T * Z + λI)^-1 * Z^T * y
    def VMA(self,e_l,e_r,k):
        start = time.time()
        z_l_vma1 = []
        z_r_vma1 = []
        
        #-------------------------------------------------------------
        # VMAモデルのパラメータを更新するための行列Zを導出
        # https://logics-of-blue.com/var%E3%83%A2%E3%83%87%E3%83%AB/ を参照(VARモデルと基本同じ(多分))
        # このfor文の段階では q x (N-d) 行列
        for i in range(self.q):
            z_l_vma0 = []
            z_r_vma0 = []
            for j in range(self.N-self.d):
                z_l_vma0.append(self.k_tEps[(j+i+2+self.s):(j+i+3+self.s)][0])
                z_r_vma0.append(self.k_xEps[(j+i+2+self.s):(j+i+3+self.s)][0])
            z_l_vma1.append(z_l_vma0)
            z_r_vma1.append(z_r_vma0)
        # p x (N-d) -> (N-d) x p (Z行列は(N-d) x p)
        z_l_vma1 = np.array(z_l_vma1).T
        z_r_vma1 = np.array(z_r_vma1).T
        # y = wx + b の線形回帰の式のbをWに融合させるために最後の列に1の列を追加
        z_l_vma1 = np.append(z_l_vma1, np.ones([z_l_vma1.shape[0],1]),axis=1)
        z_r_vma1 = np.append(z_r_vma1, np.ones([z_r_vma1.shape[0],1]),axis=1)
        #-------------------------------------------------------------

        #-----------------------------------------------------------
        # self.w_maの更新
        sigma_l_vma0 = np.matmul(z_l_vma1.T, z_l_vma1)
        sigma_r_vma0 = np.matmul(z_r_vma1.T, z_r_vma1)

        sigma_l_vma0 += 0.0000001 * np.eye(sigma_l_vma0.shape[0])
        sigma_r_vma0 += 0.0000001 * np.eye(sigma_r_vma0.shape[0])

        sigma_l_vma1 = np.matmul(z_l_vma1.T, e_l)
        sigma_r_vma1 = np.matmul(z_r_vma1.T, e_r)

        w_l_vma_buf = np.matmul(np.linalg.inv(sigma_l_vma0), sigma_l_vma1)
        w_r_vma_buf = np.matmul(np.linalg.inv(sigma_r_vma0), sigma_r_vma1)

        if k == 0:
            self.w_l_vma = np.append(self.w_l_vma, w_l_vma_buf)[np.newaxis].T 
            self.w_r_vma = np.append(self.w_r_vma, w_r_vma_buf)[np.newaxis].T 
        else:
            self.w_l_vma = np.append(self.w_l_vma, w_l_vma_buf, axis=1)
            self.w_r_vma = np.append(self.w_r_vma, w_r_vma_buf, axis=1)
        #-----------------------------------------------------------

        end_time = time.time() - start
        print("time_MA : {0}".format(end_time) + "[sec]")
        print('w_l_vma :', k)
        print(self.w_l_vma)
        print('w_r_vma :', k)
        print(self.w_r_vma)
        #pdb.set_trace()
    #------------------------------------------------------------

    #------------------------------------------------------------
    # VARIMAモデルの学習
    # 
    # y : 1 ~ N 日前の時系列データを格納した行列(ベクトル)
    # e : 1 ~ N 日前のホワイトノイズを格納した行列(ベクトル)
    def train(self):
        start_train = time.time()
        #------------------------------------------------------------
        # 各キロ程ごとの時系列データ(amount日分)を取得し、y・e 行列(ベクトル)を作成
        # 行列の掛け算を行うために[np.newaxis]をy・e行列(ベクトル)にかけている
        for k in range(self.krage_length):
            self.kData = self.tData[:,k]
            self.xData = np.array(self.explain)[:,:,k]
            self.k_tEps = self.eps_l[:,k]
            self.k_xEps = self.eps_r[:,k]

            y_l = []
            y_r = []
            e_l = []
            e_r = []
            for i in range(self.N):
                #------------------------------------------------------------
                # 1階差分の計算
                # 1番目の要素は普通に格納、それ以降は一つ前の要素から引いたものをリストに格納
                if i == 0:
                    # yリストに時系列データを格納
                    y_l.append(float(self.kData[i+1+self.s]))
                    y_r.append(float(self.kData[i+1+self.s]))
                    # eリストに時系列データを格納
                    e_l.append(self.k_tEps[i+self.s])
                    e_r.append(self.k_xEps[i+self.s])
                else:
                    # yリストに時系列データを格納                    
                    y_l.append(float(self.kData[i+1+self.s]))
                    y_r.append(float(self.kData[i+1+self.s]))
                    # 一つ前に格納したデータから引く
                    y_l[i-1] -= y_l[i]
                    y_r[i-1] -= y_r[i]
                    # eリストに時系列データを格納
                    e_l.append(self.k_tEps[i+self.s])
                    e_r.append(self.k_xEps[i+self.s])
                    # 一つ前に格納したデータから引く
                    e_l[i-1] = e_l[i-1] - e_l[i]
                    e_r[i-1] = e_r[i-1] - e_r[i]
                #------------------------------------------------------------
            # 一番後ろのデータは素のデータ(引かれてない)なので省く    
            y_l = np.array(y_l[:self.N-self.d])[np.newaxis].T
            y_r = np.array(y_r[:self.N-self.d])[np.newaxis].T
            e_l = np.array(e_l[:self.N-self.d])[np.newaxis].T
            e_r = np.array(e_r[:self.N-self.d])[np.newaxis].T
            #------------------------------------------------------------
            # VARモデルとVMAモデルの計算
            self.VAR(y_l,y_r,k)
            self.VMA(e_l,e_r,k)
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

    w_l_vma_list = []
    w_r_vma_list = []

    eps_l_list = []
    eps_r_list = []

    start_all = time.time()
    for no in range(len(fileind)):
        # Varimaクラスのインスタンス化
        varima = Varima(mytrackData.train_xData[no],mytrackData.train_tData[no])
        # Varimaモデルの学習
        varima.train()
        #--------------------------------------
        # VAR・VMAで求めた重みを出力するためにリストに格納
        w_l_var_list.append(varima.w_l_var)
        w_r_var_list.append(varima.w_r_var)
        w_l_vma_list.append(varima.w_l_vma)
        w_r_vma_list.append(varima.w_r_vma)
        eps_l_list.append(varima.eps_l)
        eps_r_list.append(varima.eps_r)
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
    
    #--------------------------------------
    # VMAで求めた重みのリストをファイル出力
    f_l_vma = open("w_l_vma_list.binaryfile","wb")
    f_r_vma = open("w_r_vma_list.binaryfile","wb")
    pickle.dump(w_l_vma_list,f_l_vma)
    pickle.dump(w_r_vma_list,f_r_vma)
    f_l_vma.close()
    f_r_vma.close()
    #--------------------------------------

    #--------------------------------------
    # VMAで使用した ε のリストをファイル出力    
    f_eps_l = open("eps_l_list.binaryfile","wb")
    f_eps_r = open("eps_r_list.binaryfile","wb")
    pickle.dump(eps_l_list,f_eps_l)
    pickle.dump(eps_r_list,f_eps_r)
    f_eps_l.close()
    f_eps_r.close()
    #--------------------------------------    
