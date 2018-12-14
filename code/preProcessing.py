
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


class trackData():
    def __init__(self,fname,trainRatio=0.8):
        # dataPath & visualPath
        self.trainPer = trainRatio

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

        self.NaN()
        x,t = self.devide()
        self.get_train_test_data(x,t)

    #-----------------------------------
    #-----------------------------------

    ## NaNの処理
    def NaN(self):

        print("Now Loading.")
        # index をキロ程と日付にする
        # self.B.set_index(['krage','date'],inplace=True)

        # # 始めと終わりのインデックス
        # sInd = self.B.index[0][0]
        # eInd = self.B.index[-1][0]

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

    def devide(self):
        xData_A = self.A.drop('hll',axis=1)
        tData_A = self.A[['date','hll']]

        xData_B = self.B.drop('hll',axis=1)
        tData_B = self.B[['date','hll']]

        xData_C = self.C.drop('hll',axis=1)
        tData_C = self.C[['date','hll']]

        xData_D = self.D.drop('hll',axis=1)
        tData_D = self.D[['date','hll']]

        xData = [xData_A,xData_B,xData_C,xData_D]
        tData = [tData_A,tData_B,tData_C,tData_D]
        return xData,tData

    def get_train_test_data(self,x,t):
        fNum = len(x)

        self.train_xData = []
        self.test_xData = []
        self.train_tData = []
        self.test_tData = []

        for no in range(fNum):#trainとtestの作成
            xNum = x[no].shape[0]
            traNum = int(xNum*self.trainPer)
            self.train_xData.append(x[no][:traNum])
            self.test_xData.append(x[no][traNum:])
            self.train_tData.append(t[no][:traNum])
            self.test_tData.append(t[no][traNum:])

    def dump_data(self):
        fileind = ['A','B','C','D']

        for no in range(len(fileind)):
            fname_xTra = "xTrain_{}.binaryfile".format(fileind[no])
            fname_xTes = "xTest_{}.binaryfile".format(fileind[no])
            fname_tTra = "tTrain_{}.binaryfile".format(fileind[no])
            fname_tTes = "tTest_{}.binaryfile".format(fileind[no])

            self.dump_file(fname_xTra, self.train_xData[no])
            self.dump_file(fname_xTes, self.test_xData[no])
            self.dump_file(fname_tTra, self.train_tData[no])
            self.dump_file(fname_tTes, self.test_tData[no])

    def dump_file(self,filename,data):
        f = open(filename,'wb')
        pickle.dump(data,f)
        f.close

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

if __name__ == "__main__":

    isWindows = True

    trackfiles= ['track_A.csv','track_B.csv','track_C.csv','track_D.csv']
    # eqpfiles = ['equipment_A.csv','equipment_B.csv','equipment_C.csv','equipment_D.csv']

    myData = trackData(trackfiles)

    myData.dump_data()
