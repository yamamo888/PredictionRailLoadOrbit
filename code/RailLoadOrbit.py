#*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:25:31 2018

@author: yu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:42:54 2018

@author: yu
"""
import pickle
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
import random
import datetime as dt
import heapq
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import datetime
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import pdb
import statsmodels.api as sm


class Preprocessing():
    def __init__(self,fname):
        # dataPath & visualPath 
        if isWindows == True:
            self.dataPath = '..\\data'
            self.visualPath = '..\\visualization'
            self.featuresPath = '..\\features'
            self.trainDataPath = 'train'
            self.testDataPath = 'test'
        else:
            self.dataPath = '../data'
            self.visualPath = '../visualization'
            self.featuresPath = '../features'
            self.trainDataPath = 'train'
            self.tetsDataPath = 'test'
        
        print("Now Loading.")
        # 各開始日
        self.sTimeAC,self.sTimeB,self.sTimeD= pd.to_datetime('2017-04-01'),pd.to_datetime('2017-04-03'),pd.to_datetime('2017-04-09')
        # 終了日
        self.eTime = pd.to_datetime('2018-03-31')
        # Testデータ開始日
        self.eTrainTime= pd.to_datetime('2018-01-01')
        
        # 全日数
        self.timeAC,self.timeB,self.timeD = (self.eTime-self.sTimeAC).days+1,(self.eTime-self.sTimeB).days+1,(self.eTime-self.sTimeD).days+1        
        # 学習データ日数(2017)
        self.TraintimeAC,self.TraintimeB,self.TraintimeD = (self.eTrainTime-self.sTimeAC).days+1,(self.eTrainTime-self.sTimeB).days+1,(self.eTrainTime-self.sTimeD).days+1
        # 評価データ(2018/1/1~2018/3/31)
        self.TesttimeAC,self.TesttimeB,self.TesttimeD = self.timeAC-self.TraintimeAC,self.timeB-self.TraintimeB,self.timeD-self.TraintimeD

        pdb.set_trace()
        # trackFilesPath
        trackApath = os.path.join(self.dataPath,fname[0]) 
        trackBpath = os.path.join(self.dataPath,fname[1])
        trackCpath = os.path.join(self.dataPath,fname[2])
        trackDpath = os.path.join(self.dataPath,fname[3])
        
        # 軌道検測csvデータ読み込み
        self.A = pd.read_csv(trackApath,',')
        print("Now Loading.trackA.csv")
        """
        self.B = pd.read_csv(trackBpath,',')
        print("Now Loading.trackB.csv")
        self.C = pd.read_csv(trackCpath,',')
        print("Now Loading.trackC.csv")
        self.D = pd.read_csv(trackDpath,',')
        print("Now Loading.trackD.csv")
        """
        # column ：日本語名->英語名
        self.A.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        """
        self.B.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        self.C.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        self.D.rename(columns={'キロ程':'krage','高低左':'hll','高低右':'hlr','通り左':'sl','通り右':'sr','水準':'level','軌間':'gauge','速度':'v'},inplace=True)
        """

        # date列：object型->datetime型
        self.A['date'] = pd.to_datetime(self.A['date']) 
        """
        self.B['date'] = pd.to_datetime(self.B['date']) 
        self.C['date'] = pd.to_datetime(self.C['date']) 
        self.D['date'] = pd.to_datetime(self.D['date'])
        """
        # キロ程の初めと終わりのindex取得
        self.ksIndA = int(self.A['krage'].head(1))
        """
        self.ksIndB = int(self.B['krage'].head(1))
        self.ksIndC = int(self.C['krage'].head(1))
        self.ksIndD = int(self.D['krage'].head(1))
        """
        self.keIndA = int(self.A['krage'].tail(1))
        """
        self.keIndB = int(self.B['krage'].tail(1))
        self.keIndC = int(self.C['krage'].tail(1))
        self.keIndD = int(self.D['krage'].tail(1))
        """
        # 目的変数
        self.targetName = 'hll'
        

        
    #-----------------------------------
    #-----------------------------------
    
    ## NaNの処理
    def NaN(self):
        
        print("NaN..")
        
        # index をキロ程と日付にする
        self.A.set_index(['krage','date'],inplace=True)
        """
        self.B.set_index(['krage','date'],inplace=True)
        self.C.set_index(['krage','date'],inplace=True)
        self.D.set_index(['krage','date'],inplace=True)
        """
        
        #pdb.set_trace()
        """
        row = 10
        columns = 10
        graphs = row*columns
        axes = []
        fig = plt.figure(figsize=(row,columns))
        Ind = np.arange(self.ksIndA,self.keIndA,1)
        for i in range(1,graphs+1):
            axes.append(fig.add_subplot(row,columns,i))
            axes[i-1].plot(np.arange(365),self.A['hll'][Ind[i]])
        """
        Ind = np.arange(self.ksIndA,self.keIndA,1)
        for i in range(10):
            plt.plot(np.arange(365),self.A['hll'][Ind[i]])
    
            plt.savefig('Adata_{}'.format(i))
            plt.close() 
     
        # すべてのNaN平均補完
        aveA = self.A['hll'].mean()
        stdA = self.A['hll'].std()
        sumNullA = self.A['hll'].isnull().sum()
        randA = np.random.randint(aveA-stdA,aveA+stdA,size=sumNullA)
        self.A['hll'][np.isnan(self.A['hll'])] = randA
        """
        aveB = self.B['hll'].mean()
        stdB = self.B['hll'].std()
        sumNullB = self.B['hll'].isnull().sum()
        randB = np.random.randint(aveB-stdB,aveB+stdB,size=sumNullB)
        self.B['hll'][np.isnan(self.B['hll'])] = randB
        
        aveC= self.C['hll'].mean()
        stdC = self.C['hll'].std()
        sumNullC = self.C['hll'].isnull().sum()
        randC = np.random.randint(aveC-stdC,aveC+stdC,size=sumNullC)
        self.C['hll'][np.isnan(self.C['hll'])] = randC
        
        aveD = self.D['hll'].mean()
        stdD = self.D['hll'].std()
        sumNullD = self.D['hll'].isnull().sum()
        randD = np.random.randint(aveD-stdD,aveD+stdD,size=sumNullD)
        self.D['hll'][np.isnan(self.D['hll'])] = randD
        """
        
        
    #-----------------------------------
    #-----------------------------------
            
    ## 設備要因        
    def equipmentData(self,fname):
    
        # equipmentFilesPath
        equipmentApath = os.path.join(self.dataPath,fname[0])
        equipmentBpath = os.path.join(self.dataPath,fname[1])
        equipmentCpath = os.path.join(self.dataPath,fname[2])
        equipmentDpath = os.path.join(self.dataPath,fname[3])
    
        
        # 設備台帳データ
        self.equipmentA = pd.read_csv(equipmentApath,',')
        print("Now Loading.....")
        """
        self.equipmentB = pd.read_csv(equipmentBpath,',')
        print("Now Loading.....")
        self.equipmentC = pd.read_csv(equipmentCpath,',')
        print("Now Loading.....")
        self.equipmentD = pd.read_csv(equipmentDpath,',')
        print("Now Loading.....")
        """
        #pdb.set_trace()

    def SaveTrainData(self,fname='train.pkl'):
        # 学習データ(2017/4/1~2018/1/31)：高低左
        trainA = self.A[self.targetName] 
        """
        trainB = self.B[self.targetName]
        trainC = self.C[self.targetName]
        trainD = self.D[self.targetName]
        
        # 測定された日数
        krageA = np.arange(self.ksIndA,self.keIndA+1)
        krageB = np.arange(self.ksIndB,self.keIndB+1)
        krageC = np.arange(self.ksIndC,self.keIndC+1)
        krageD = np.arange(self.ksIndD,self.keIndD+1)
        """
        pdb.set_trace()


        # TrackA
        flag = False 
        for kInd in np.arange(self.keIndA-self.ksIndA):
            tmp = trainA[(krageA[kInd])].values

            if not flag:
                self.tmpA = tmp[:,np.newaxis]
                flag = True
            else:
                self.tmpA = np.concatenate((self.tmpA,tmp[:,np.newaxis]),1)
        
        print("SaveTrainA") 
        """
        # TrackB
        flag = False 
        for kInd in np.arange(self.keIndB-self.ksIndB):
            tmp = trainB[(krageB[kInd])].values

            if not flag:
                self.tmpB = tmp[:,np.newaxis]
                flag = True
            else:
                self.tmpB = np.concatenate((self.tmpB,tmp[:,np.newaxis]),1)
        
        # TrackC
        flag = False 
        for kInd in np.arange(self.keIndC-self.ksIndC):
            tmp = trainC[(krageC[kInd])].values

            if not flag:
                self.tmpC = tmp[:,np.newaxis]
                flag = True
            else:
                self.tmpC = np.concatenate((self.tmpC,tmp[:,np.newaxis]),1)
        
        # TrackD
        flag = False 
        for kInd in np.arange(self.keIndD-self.ksIndD):
            tmp = trainB[(krageD[kInd])].values

            if not flag:
                self.tmpD = tmp[:,np.newaxis]
                flag = True
            else:
                self.tmpD = np.concatenate((self.tmpD,tmp[:,np.newaxis]),1)
        """
        # 全データから学習データ取得
        self.TrainA = self.tmpA[:self.TraintimeAC]
        #self.TrainB = self.tmpB[:self.TraintimeB]
        #self.TrainC = self.tmpC[:self.TraintimeAC]
        #self.TrainD = self.tmpD[:self.TraintimeD]
        
        #pdb.set_trace()
        # 学習データ保存
        fullPath = os.path.join(self.featuresPath,fname)
        with open(fullPath,'wb') as fp:
            pickle.dump(self.TrainA,fp)
            #pickle.dump(self.TrainB,fp)
            #pickle.dump(self.TrainC,fp)
            #pickle.dump(self.TrainD,fp)


    def SaveTestData(self,fname='test.pkl'):
        
        print("SaveTest") 
        # 評価データ(2018/2/1~2018/3/31):終わり２か月で評価
        self.TestA = self.tmpA[self.TraintimeAC:] 
        #self.TestB = self.tmpB[self.TraintimeB:]
        #self.TestC = self.tmpC[self.TraintimeAC:]
        #self.TestD = self.tmpD[self.TraintimeD:]
        
        fullPath = os.path.join(self.featuresPath,fname)
        with open(fullPath,'wb') as fp:
            pickle.dump(self.TestA,fp)
            #pickle.dump(self.TestB,fp)
            #pickle.dump(self.TestC,fp)
            #pickle.dump(self.TestD,fp)
        #pdb.set_trace() 

class Training():
    def __init__(self):
        
        self.features = "../features"
        
        """
        # MydatapickleName
        trainpname = 'train.pkl'
        testpname = 'test.pkl'
        # MyTrain,TestpickleデータPath
        #TrainfullPath = os.path.join(self.features,trainpname)
        #TestfullPath = os.path.join(self.features,testpname)
        """
        # NaN補完データpickle
        trainApname = 'track_tTrain_A.binaryfile'
        testApname = 'track_tTest_A.binaryfile'
        trainBpname = 'track_tTrain_B.binaryfile'
        testBpname = 'track_tTest_B.binaryfile'
        trainCpname = 'track_tTrain_C.binaryfile'
        testCpname = 'track_tTest_C.binaryfile'
        trainDpname = 'track_tTrain_D.binaryfile'
        testDpname = 'track_tTest_D.binaryfile'
        # NaN補完データPath
        TrainAfullPath = os.path.join(self.features,trainApname)
        TestAfullPath = os.path.join(self.features,testApname)
        TrainBfullPath = os.path.join(self.features,trainBpname)
        TestBfullPath = os.path.join(self.features,testBpname)
        TrainCfullPath = os.path.join(self.features,trainCpname)
        TestCfullPath = os.path.join(self.features,testCpname)
        TrainDfullPath = os.path.join(self.features,trainDpname)
        TestDfullPath = os.path.join(self.features,testDpname)
         
        """ 
        # MyTrain,MyTestデータ取得
        with open(TrainfullPath,'rb') as fp:
            self.TrainA = pickle.load(fp)
            #self.TrainB = pickle.load(fp)
            #self.TrainC = pickle.load(fp)
            #self.TrainD = pickle.load(fp)
        
        # 縦軸：時間　横軸：キロ程
        self.TrainA = self.TrainA.T
        #self.TrainB = self.TrainB.T
        #self.TrainC = self.TrainC.T
        #self.TrainD = self.TrainD.T

        with open(TestfullPath,'rb') as fp:
            self.TestA = pickle.load(fp)
            #self.TestB = pickle.load(fp)
            #self.TestC = pickle.load(fp)
            #self.TestD = pickle.load(fp)
        
        
        # 縦軸：時間　横軸：キロ程
        self.TestA = self.TestA.T
        #self.TestB = self.TestB.T
        #self.TestC = self.TestC.T
        #self.TestD = self.TestD.T
        """

        # NaN補完データ
        with open(TrainAfullPath,'rb') as fp:
            self.trainAX = pickle.load(fp)
        with open(TrainBfullPath,'rb') as fp:
            self.trainBX = pickle.load(fp)
        with open(TrainCfullPath,'rb') as fp:
            self.trainCX = pickle.load(fp)
        with open(TrainDfullPath,'rb') as fp:
            self.trainDX = pickle.load(fp)
        
        with open(TestAfullPath,'rb') as fp:
            self.testAX = pickle.load(fp)
        with open(TestBfullPath,'rb') as fp:
            self.testBX = pickle.load(fp)
        with open(TestCfullPath,'rb') as fp:
            self.testCX = pickle.load(fp)
        with open(TestDfullPath,'rb') as fp:
            self.testDX = pickle.load(fp)


        """
        pdb.set_trace()
        self.trainData = np.zeros([self.TrainA.shape[0],self.TrainA.shape[1]])
        for trkrange in range(self.TrainA.shape[0]):
            self.trainData[trkrange] = self.TrainA[trkrange]
        
        self.testData = np.zeros([self.TestA.shape[0],self.TestA.shape[1]])
        for tekrange in range(self.TestA.shape[0]):
            self.testData[tekrange] = self.TestA[tekrange]
        """
        
        # plot
        """ 
        row = 10
        columns = 10
        graphs = row*columns
        axes = []
        fig = plt.figure(figsize=(row,columns))
        for i in range(1,graphs+1):
            axes.append(fig.add_subplot(row,columns,i))
            axes[i-1].plot(np.arange(self.trainAX.shape[0]),self.trainAX.T[i])
        
        
        plt.savefig('trainA')
        
        row = 10
        columns = 10
        graphs = row*columns
        axes = []
        fig = plt.figure(figsize=(row,columns))
        for i in range(1,graphs+1):
            axes.append(fig.add_subplot(row,columns,i))
            axes[i-1].plot(np.arange(self.testAX.shape[0]),self.testAX.T[i])
        
        
        plt.savefig('testA')
        
        row = 10
        columns = 10
        graphs = row*columns
        axes = []
        fig = plt.figure(figsize=(row,columns))
        for i in range(1,graphs+1):
            axes.append(fig.add_subplot(row,columns,i))
            axes[i-1].plot(np.arange(self.trainBX.shape[0]),self.trainBX.T[i])
        
        plt.savefig('trainB')
        
        row = 10
        columns = 10
        graphs = row*columns
        axes = []
        fig = plt.figure(figsize=(row,columns))
        for i in range(1,graphs+1):
            axes.append(fig.add_subplot(row,columns,i))
            axes[i-1].plot(np.arange(self.trainCX.shape[0]),self.trainCX.T[i])
        
        plt.savefig('trainC')
        
        row = 10
        columns = 10
        graphs = row*columns
        axes = []
        fig = plt.figure(figsize=(row,columns))
        for i in range(1,graphs+1):
            axes.append(fig.add_subplot(row,columns,i))
            axes[i-1].plot(np.arange(self.trainDX.shape[0]),self.trainDX.T[i])
        
        plt.savefig('trainD')
        """
        
        pdb.set_trace() 
        
        

        # RNN 用データ整形
        # TrainData
        trainAX_rnn = np.array(list(self.trainAX)*self.rnnNum)
        trainBX_rnn = np.array(list(self.trainBX)*self.rnnNum)
        trainCX_rnn = np.array(list(self.trainCX)*self.rnnNum)
        trainDX_rnn = np.array(list(self.trainDX)*self.rnnNum)
        # TestData
        testAX_rnn = np.array(list(self.testAX)*self.rnnNum)
        testBX_rnn = np.array(list(self.testBX)*self.rnnNum)
        testCX_rnn = np.array(list(self.testCX)*self.rnnNum)
        testDX_rnn = np.array(list(self.testDX)*self.rnnNum)


        



        # trainData日数
        self.trainAdays = self.trainA.shape[0]
        self.trainBdays = self.trainB.shape[0]
        self.trainCdays = self.trainC.shape[0]
        self.trainDdays = self.trainD.shape[0]
        
        # testData日数
        self.testdays = 89 
        
        # キロ程
        self.Akrage = self.trainA.shape[1]
        self.Bkrage = self.trainB.shape[1]
        self.Ckrage = self.trainC.shape[1]
        self.Dkrage = self.trainD.shape[1]
        
        # Params
        self.numLayer = 2
        self.numSteps = 1
        self.hiddenLayer = 200
        self.batchSize = 276
        self.numDimensions = 1 
        self.aveNum = 5
        self.Rnum = 276
        self.hiddenRnum = 80
        self.inputRnum = 250 
        self.outputRnum = self.testdays
        batchCnt = 0
        self.batchRCnt = 0
        batchSize = 10
        self.batchRSize = 500
        self.rnnNum = 10
        
    

################# RNN #############################
##################################################

    def nextBatch(self,data,batchSize,numSteps,numDimensions):
        # epoch数指定
        epoch = data.size // (batchSize*numSteps*numDimensions)
        # dataを分割
        data = np.lib.stride_tricks.as_strided(data,shape=(epoch,batchSize,numSteps+1,numDimensions),strides=(4*batchSize*numSteps*numDimensions,4*numSteps*numDimensions,4*numDimensions,4),writeable=False)
    

        return data[0,:,:-1][np.newaxis,:,:],data[:,:,1:]

    def weight_variable(self,name,shape):
        return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    
    def bias_variable(self,name,shape):
        return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
    
    def fc_relu(self,inputs,w,b,keepProb):
        relu = tf.matmul(inputs,w)+b
        relu = tf.nn.dropout(relu,keepProb)
        relu = tf.nn.relu(relu)
        return relu
    
    def fc(self,inputs,w,b,keepProb):
        fc = tf.matmul(inputs,w)+b
        fc = tf.nn.dropout(fc,keepProb)
        return fc

    def RNN(self,x,reuse=False):
        with tf.variable_scope('RNN') as scope:
            keepProb = 1.0
            if reuse:
                keepProb = 1.0
                scope.reuse_variables()

            x = tf.reshape(x,[-1,numDimensions])
            inputW = self.weight_variable('inputW',[self.numDimensions,self.hiddenLayer])
            inputbias = self.bias_variable('inputbias',[self.hiddenLayer])
            
            x = self.fc(x,inputW,inputbias,keepProb)
            x = tf.reshape(x,[self.batchSize,self.numSteps,self.hiddenLayer])

            # LSTM
            cell = tf.contrib.rnn.LSTMBlockCell(self.hiddenLayer,forget_bias=0.0)
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.numLayer)],state_is_tuple=True)
            init = cell.zero_state(self.batchSize,tf.float32)

            output,state = tf.nn.dynamic_rnn(cell,x,initial_state=init)

            output = tf.reshape(output,[-1,self.hiddenLayer])
            outputW = self.weight_variable('outputW',[self.hiddenLayer,1])
            outputbias = self.bias_variable('outputbias',[1])
            output = self.fc(output,outputW,outputbias,keepProb)
            output = tf.reshape(output,[self.batchSize,self.numSteps,1])
            
            return output
    
    def nextRBatch(self,data):
        
        batchX = data[:,:self.trainDay]
        batchY = data[:,self.trainDay:]

        srInd = self.batchRSize * self.batchRCnt
        erInd = srInd +  self.batchRSize

        batchX = batchX[srInd:erInd]
        batchY = batchY[srInd:erInd]
        
        if erInd+self.batchRSize > self.TrainRA.shape[0]:
            self.batchRCnt = 0
        else:
            self.batchRCnt += 1
        
        return batchX,batchY

    def RegressionNN(self,xr,reuse=False):
        with tf.variable_scope('RNN') as scope:
            keepProb = 1.0
            if reuse:
                keepProb = 1.0
                scope.reuse_variables()


        w1 = self.weight_variable('w1',[self.inputRnum,self.hiddenRnum])
        b1 = self.bias_variable('b1',[self.hiddenRnum])

        fc1 = self.fc_relu(xr,w1,b1,keepProb)

        w2 = self.weight_variable('w2',[self.hiddenRnum,self.outputRnum])
        b2 = self.bias_variable('b2',[self.outputRnum])

        y = self.fc(fc1,w2,b2,keepProb)

        return y
        



    def Learning(self,rnn_op,rnn_test_op,regression_op):
        #pdb.set_trace()
        loss = tf.reduce_mean(tf.square(rnn_op[:,-1,:]-inputStrided[:,-1,:]))
        trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
        #trainer = tf.train.GradientDescentOptimizer(1e-3).minimizer(loss)
        loss_reg = tf.reduce_mean(tf.abs(regression_op-y_))
        trainer_reg = tf.train.AdamOptimizer(1e-3).minimize(loss_reg)
    
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        rnnbox = []
        batchCnt = 0
        
        predMA = np.zeros([])
        # 移動平均
        for i in range():
            aveA = np.convolve(self.trainData[sInd:eInd],np.ones(self.aveNum)/np.float(self.aveNum),'same')
        
        
        #　線形回帰 
        for i in range():

            _,TrainRloss,Regression = sess.run([trainer_reg,loss_reg,regression_op],feed_dict={xr:batchX,y_:batchY})
        
        # RNN
        for i in range(9):
            #pdb.set_trace()
            for j in range(self.Tr.shape[0]):
                #正規化
                #self.trainData = (self.Tr[j]-np.min(self.Tr[j]))/(np.max(self.Tr[j])-np.min(self.Tr[j]))
                self.trainData = self.Tr[j]
                trainInput, trainInputStrided = self.nextBatch(self.trainData,self.batchSize,self.numSteps,self.numDimensions)
                batchX,batchY = self.nextRBatch(self.TrainRA)


                pdb.set_trace()
                _,Trainloss,RNNCell = sess.run([trainer,loss,rnn_op],feed_dict={x:trainInput[i].astype(np.float32),inputStrided:trainInputStrided[i].astype(np.float32)})
                
                pdb.set_trace()
                sInd = self.batchSize*j
                eInd = sInd + self.batchSize
                

                # plot
                """
                plt.plot(np.arange(self.trainData.shape[0]),self.trainData,color='k')
                plt.plot(np.arange(self.trainData.shape[0]),aveA,'y')
                plt.plot(np.arange(self.trainData.shape[0]),aveA*1.5,'m')
                plt.plot(np.arange(self.trainData.shape[0]),np.array(list(RNNCell[:,-1,-1])),color='c')
                """
                plt.plot(self.trainData[:self.batchSize],color='k')
                plt.plot(aveA,'y')
                plt.plot(aveA*1.5,'m')
                plt.plot(np.array(list(RNNCell[:,-1,-1])),color='c')
                plt.plot(Regression[0,:])
                plt.savefig('rnn{}'.format(i))
                plt.close()
                pdb.set_trace()


                print("ite: %d, Trainloss: %f"%(i,Trainloss))
                #print(RNNcell[:10])
                """
                if i % 5 == 0:
                    
                    testInput, testInputStrided = self.nextBatch(self.testData,self.batchSize,self.numSteps,self.numDimensions)
                    
                    Testloss = sess.run(loss,feed_dict={x:testInput[i],inputStrided:testInputStrided})
                    
                    print("ite %d, Testloss: %f"%(i,Testloss))"""


class Predict():
    def __init__(self,fname):
        """
        # 教師データ
        with open(,'rb') as fp
            self.TestA = pickle.load(fp)
            #self.TestB = pickle.load(fp)
            #self.TestC = pickle.load(fp)
            #self.TestD = pickle.load(fp)"""

    # ユークリッド距離    
    def distance(self,p0,p1):
        return np.sum((p0-p1)**2)
    
    def kNN(self):
        """
        # ARモデル
        predAR

        # MAモデル
        predMA

        # RNNモデル
        predRNN

        # 線形モデル
        predLiner
        
        # 教師データ
        trueData = 
        """
        # 真値との距離計算
        distAR = distance(trueData,predAR)
        distMA = distance(trueData,predMA)
        distRNN = distance(trueData,predRNN)
        distLiner = distance(trueData,predLiner)
        
        # すべてのデータを1つにまとめる
        allData = np.array([distAR,distMA,distRNN,distLiner])
        # 各キロ程の最小インデックス(データに対応)取得
        minInd = np.argmin(allData,axis=0)
        # キロ程ごとのデータ
        predANum = maxInd.shape[0] 
        # 最終的な予測データ
        predAData = np.zeros([predANum,1])
        for i in np.arange(maxInd):
            if maxInd[i] == 0:
                predAData[i] = predAR[i]
            elif maxInd[i] == 1:
                predAData[i] = predMA[i]
            elif maxInd[i] == 2:
                predAData[i] = predRNN[i]
            elif maxInd[i] == 3:
                predAData[i] =  predLiner[i]
        
        



if __name__ == '__main__':
    isWindows = False
    # files
    trackfiles= ['track_A.csv','track_B.csv','track_C.csv','track_D.csv']
    eqpfiles = ['equipment_A.csv','equipment_B.csv','equipment_C.csv','equipment_D.csv']

    #mytrackData = Preprocessing(trackfiles)
    #mytrackData.NaN()
    #mytrackData.SaveTrainData(fname='train.pkl')
    #mytrackData.SaveTestData(fname='test.pkl')
    #mytrackData.equipmentData(eqpfiles)
    
    ## Learning
    training = Training()
    numSteps = training.numSteps
    numDimensions = training.numDimensions
    batchSize = training.batchSize
    inputRDimensions = training.inputRnum
    outputRDimensions = training.outputRnum
    
    # placeholder
    x = tf.placeholder(tf.float32,[None,numSteps,numDimensions])
    xr = tf.placeholder(tf.float32,[None,inputRDimensions])
    y_ = tf.placeholder(tf.float32,[None,outputRDimensions])
    inputStrided = tf.placeholder(tf.float32,[batchSize,numSteps,numDimensions])
    
    RNNTrain = training.RNN(x)
    RNNTest = training.RNN(x,reuse=True)
    RTrain = training.RegressionNN(xr)
    training.Learning(RNNTrain,RNNTest,RTrain)


        
        
        
