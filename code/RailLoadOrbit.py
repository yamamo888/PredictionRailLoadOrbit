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

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pylab as plt

import datetime

import pdb


class trackData():
    def __init__(self,fname):
        # dataPath & visualPath 
        if isWindows == True:
            self.dataPath = '..\\data'
            self.visualPath = '..\\visualization'
        else:
            self.dataPath = '../data'
            self.visualPath = '../visualization'
        
        print("Now Loading.")
        # 各開始日
        self.sTimeAC,self.sTimeB,self.sTimeD= pd.to_datetime('2017-04-01'),pd.to_datetime('2017-04-03'),pd.to_datetime('2017-04-09')
        # 終了日
        eTime = pd.to_datetime('2018-03-31')
        # 各学習データ期間
        self.timeAC,self.timeB,self.timeD = (eTime-self.sTimeAC).days+1,(eTime-self.sTimeB).days+1,(eTime-self.sTimeD).days+1        
        

        # trackFilesPath
        trackApath = os.path.join(self.dataPath,fname[0]) 
        trackBpath = os.path.join(self.dataPath,fname[1])
        trackCpath = os.path.join(self.dataPath,fname[2])
        trackDpath = os.path.join(self.dataPath,fname[3])
        
        # 軌道検測csvデータ読み込み
        self.A = pd.read_csv(trackApath,',')
        print("Now Loading..")
        self.B = pd.read_csv(trackBpath,',')
        print("Now Loading...")
        self.C = pd.read_csv(trackCpath,',')
        print("Now Loading....")
        self.D = pd.read_csv(trackDpath,',')
        print("Now Loading.....")
        
        # column ：日本語名->英語名
        self.A.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        self.B.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        self.C.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        self.D.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        
        # date列：object型->datetime型
        self.A['date'] = pd.to_datetime(self.A['date']) 
        self.B['date'] = pd.to_datetime(self.B['date']) 
        self.C['date'] = pd.to_datetime(self.C['date']) 
        self.D['date'] = pd.to_datetime(self.D['date'])
        
    #-----------------------------------
    #-----------------------------------
    
    ## NaNの処理
    def NaN(self):
        
        print("Now Loading.")        
        # index をキロ程と日付にする
        self.B.set_index(['krage','date'],inplace=True)
        
        # 始めと終わりのインデックス
        sInd = self.B.index[0][0]
        eInd = self.B.index[-1][0]
        
        # NaN:線形補完
        self.A.interpolate(method='linear',limit_direction='both',inplace=True)
        print("Now Loading..")         
        self.B.interpolate(method='linear',limit_direction='both',inplace=True)
        print("Now Loading...")         
        self.C.interpolate(method='linear',limit_direction='both',inplace=True)
        print("Now Loading....")         
        self.D.interpolate(method='linear',limit_direction='both',inplace=True)

        """
        # NaN:時系列を考慮した補完(未実装)
        flag = False
        for kInd in np.arange(sInd,eInd):
            
            tmp = self.B.loc[kInd]
            tmp.interpolate(mehod='time',limit_direction='both',inplace=True)
            
            if not flag:
                self.trackB = tmp
                flag = True
            else:
                pd.concat([self.trackB,tmp])"""
        
    #-----------------------------------
    #-----------------------------------
        
        
## 設備要因        
class equipmentData():
    def __init__(self,fname):
    
        # equipmentFilesPath
        equipmentApath = os.path.join(self.dataPath,fname[4])
        equipmentBpath = os.path.join(self.dataPath,fname[5])
        equipmentCpath = os.path.join(self.dataPath,fname[6])
        equipmentDpath = os.path.join(self.dataPath,fname[7])
    
        
        # 設備台帳データ
        self.equipmentA = pd.read_csv(equipmentApath,',')
        self.equipmentB = pd.read_csv(equipmentBpath,',')
        self.equipmentC = pd.read_csv(equipmentCpath,',')
        self.equipmentD = pd.read_csv(equipmentDpath,',')
    
class Ar():
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
        #z1 = np.empty((self.N*t.shape[0],0))
        z1 = []
        for i in range(self.p):
            z0 = []
            for j in range(self.N):
                date = np.append(date, (t['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                z0 = np.append(z0, self.t[self.t['date'] == date[-1]]['hll'])
                #date.append((t['date'][-1:] - datetime.timedelta(days=j+i+2)).astype(str))
                pdb.set_trace()
                #z0.append(self.t[self.t['date'] == date[-1]]['hll'])
            #pdb.set_trace()
            z0 = z0[np.newaxis].T
            z1 = np.append(z1, z0,axis=1)
        z1 = np.array(z1).reshape(1395300, 50)
        #pdb.set_trace()
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


        
if __name__ == '__main__':
    isWindows = False
    # files
    trackfiles= ['track_A.csv','track_B.csv','track_C.csv','track_D.csv']
    eqpfiles = ['equipment_A.csv','equipment_B.csv','equipment_C.csv','equipment_D.csv']

    mytrackData = trackData(trackfiles)
    mytrackData.NaN()
    #pdb.set_trace()
    xData = mytrackData.A.drop('hll',axis=1)
    tData = mytrackData.A[['date','hll']]
    
    ar = Ar(xData,tData)
    
    T = tData[tData['date'] == '2018-03-31']
    ar.train()
    y = ar.predict(T)
    print(y)
    loss = ar.loss(T)
    print(loss)
    #pdb.set_trace()


    

    #myequipmentData = equipmentData(eqpfiles)


        
        
        
