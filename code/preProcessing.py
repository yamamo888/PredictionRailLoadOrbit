import datetime as dt
import pandas as pd
import numpy as np
import os
#import matplotlib.pylab as plt   #折れ線グラフを作るやつ
import matplotlib.pyplot as plt


#-------------------
# pre_processingクラスの定義定義始まり
class pre_processing:
    dataPath = 'data'  # データのフォルダ名

    #------------------------------------
    # CSVファイルの読み込み
    def __init__(self):
        # trainデータの割合
        self.trainPer = 0.8

        # 軌道検測データの読み込み
        self.track = {}
        for no in ['A', 'B', 'C', 'D']:
            self.track[no] = pd.read_csv(os.path.join(self.dataPath, "track_{}.csv".format(no)), parse_dates=["date"])

        """とりあえずまだ使わない
        # 設備台帳データの読み込み
        self.equipment = {}
        for no in ['A', 'B', 'C', 'D']:
            self.equipment[no] = pd.read_csv(os.path.join(self.dataPath, "equipment_{}.csv".format(no)))
        """
    #------------------------------------

    #------------------------------------
    # 欠損データを補完する
    # 平均からばらつきを考慮して補完する
    def complement(self, data, column):
        ave = data[column].mean()  #平均値
        std = data[column].std()  #標準偏差
        cnt_null = data[column].isnull().sum()  #欠損データの総数

        # 正規分布に従うとし標準偏差の範囲内でランダムに数字を作る
        rnd = np.random.randint(ave - std, ave + std, size=cnt_null)

        data[column][np.isnan(data[column])] = rnd
    #------------------------------------

    #------------------------------------
    # 説明変数と目的変数に分ける
    def divide(self, data):
        self.complement(data, "高低左")
        self.complement(data, "高低右")
        self.complement(data, "通り左")
        self.complement(data, "通り右")
        self.complement(data, "水準")
        self.complement(data, "軌間")
        self.complement(data, "速度")

        # 目的変数である高低左を削除する
        x = data.drop(["高低左"], axis=1).values
        t = data[["高低左"]].values

        return x, t
    #------------------------------------

    #------------------------------------
    # 今のところはtrackデータのみ使う
    # trainデータを読み込む
    def get_train_data(self, no):
        trainInd = {}
        x = {}
        t = {}
        xTrain = {}
        tTrain = {}

        x[no], t[no] = self.divide(self.track[no])

        trainInd[no] = int(len(self.track[no])*self.trainPer)
        xTrain[no] = x[no][:trainInd[no]]
        tTrain[no] = t[no][:trainInd[no]]

        return xTrain[no], tTrain[no]
    #------------------------------------

    #------------------------------------
    # testデータを読み込む
    def get_test_data(self, no):
        trainInd = {}
        x = {}
        t = {}
        xTest = {}
        tTest = {}

        x[no], t[no] = self.divide(self.track[no])

        trainInd[no] = int(len(self.track[no])*self.trainPer)
        xTest[no] = x[no][trainInd[no]:]
        tTest[no] = t[no][trainInd[no]:]

        return xTest[no], tTest[no]
    #------------------------------------

    #------------------------------------
    # ファイル出力を行う
    def file_output(self, fname, data):
        fullpath = os.path.join(self.dataPath, fname)

        f = open(fullpath)
        with open(fname, mode='w') as f:
            file.writelines(data)

        f.close()
    #------------------------------------
# pre_processingクラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
    xTrain= {}
    tTrain = {}
    xTest = {}
    tTest = {}

    myData = pre_processing()

    for no in ['A', 'B', 'C', 'D']:
        # trainデータについて
        xTrain[no], tTrain[no] = myData.get_train_data(no)
        # ファイル出力
        # 説明変数
        fname = "xTrain_{}.txt".format(no)
        myData.file_output(fname, xTrain[no])
        # 目的変数
        fname = "tTrain_{}.txt".format(no)
        myData.file_output(fname, tTrain[no])

        # testデータについて
        xTest[no], tTest[no] = myData.get_test_data(no)
        # ファイル出力
        # 説明変数
        fname = "xTest_{}.txt".format(no)
        myData.file_output(fname, xTest[no])
        # 目的変数
        fname = "tTest_{}.txt".format(no)
        myData.file_output(fname, tTest[no])

#メインの終わり
#-------------------
