import datetime as dt
import pandas as pd
import numpy as np
import math
import os
#import matplotlib.pylab as plt   #折れ線グラフを作るやつ
import matplotlib.pyplot as plt
import pdb


#-------------------
# pre_processingクラスの定義定義始まり
class pre_processing:
	dataPath = 'data' # データが格納させているフォルダ名

	#------------------------------------
	# CSVファイルの読み込み
	def __init__(self):
		# trainデータの割合
		self.trainPer = 0.8

		# 軌道検測データの読み込み
		self.track = {}
		for no in ['A', 'B', 'C', 'D']:
			self.track[no] = pd.read_csv(os.path.join(self.dataPath, "track_{}.csv".format(no)), parse_dates=["date"])

		# 設備台帳データの読み込み
		self.equipment = {}
		for no in ['A', 'B', 'C', 'D']:
			self.equipment[no] = pd.read_csv(os.path.join(self.dataPath, "equipment_{}.csv".format(no)))
	#------------------------------------

	#------------------------------------
	# 値があるかないかの行列の形に変換
	# ガウシアンカーネルと同じshapeで、中心に欠損値がある行列を取得したい
	def shape_matrix(self, data, target):
		return data.groupby(["date", "キロ程"]).max()[target].unstack().values
	#------------------------------------

	#------------------------------------
	# ガウシアンカーネルを作成
	def gaussian_kernel(self, size):
		if size % 2 == 0:
			print('kernel size should be odd')
			return

		sigma = (size - 1) / 2

		# [0,size] -> [-sigma,sigma] にずらす
		x = y = np.arange(0, size) - sigma
		X, Y = np.meshgrid(x, y)

		gauss = np.exp(-(np.square(X) + np.square(Y)) / (2 * np.square(sigma)) / (2 * np.pi * np.square(sigma)))

		# 総和が1になるように
		kernel = gauss / np.sum(gauss)

		return kernel
	#------------------------------------

	#------------------------------------
	# 欠損データを補完する
	# 平均からばらつきを考慮して補完する
	def ave_complement(self, data, column):
		ave = data[column].mean() #平均値
		std = data[column].std() #標準偏差
		cnt_null = data[column].isnull().sum() #欠損データの総数

		# 正規分布に従うとし標準偏差の範囲内でランダムに数字を作る
		rnd = np.random.randint(ave - std, ave + std, size=cnt_null)

		data[column][np.isnan(data[column])] = rnd
	#------------------------------------

	#------------------------------------
	# 欠損データを補完する
	# ガウシアンカーネルと協調フィルタリングの考え方を用いる
	"""まだできていない
	def gauss_complement(self, data, index):
		# ガウシアンカーネルを作成
		gauss = self.gaussian_kernel(9)
		value =

		return value"""
	#------------------------------------

	#------------------------------------
	# 欠損値のインデックスを得る
	def get_index(self, data):
		# NaNのインデックスリスト
		list_nan = np.array([])
		list_del = np.array([]) #消去するNanのリスト
		list_fill = np.array([]) #補完するNanのリスト

		# 高低左のデータに対して処理
		nan_mat = self.shape_matrix(data, "高低左")
		#pdb.set_trace()
		for i in range(nan_mat.shape[0]):
			if(math.isnan(nan_mat[i][0]) == True):
					#pdb.set_trace()
					list_nan=np.append(list_nan,int(i))
			elif(math.isnan(nan_mat[i][0]) == False):
				if(len(list_nan) >= 10):
					list_del=np.append(list_del,list_nan)
					list_nan=np.array([])
				elif(0 < len(list_nan) < 10):
					list_fill=np.append(list_fill,list_nan)
					list_nan=np.array([])

		return list_del, list_fill
	#------------------------------------

	#------------------------------------
	# 欠損値に対する処理を行う
	def missing_values(self, data):
		newData = data
		del_,fill = self.get_index(data)
		#pdb.set_trace()
		# 削除
		for i in range(len(del_)):
			newData = np.delete(newData, del_[i])
		"""まだできていない
		# 補完
		for j in range(len(fill)):
			newData = self.gauss_complement(newData, fill[i])
		"""
		return newData
	#------------------------------------

	#------------------------------------
	# 説明変数と目的変数に分ける
	# まだこれを使える状態にはなっていない
	def divide_track(self, data):
		mat = self.missing_values(data)

		# 目的変数は高低左
		
		t = mat[:,0]

		return x, t
	#------------------------------------

	#------------------------------------
	def divide_equipment(self, data):
		x = data.drop(["橋りょう"], axis=1).values
		t = data[["橋りょう"]].values

		return x, t
	#------------------------------------

	#------------------------------------
	# 今のところはtrackデータのみ使う
	# trainデータを読み込む
	def get_train_data(self, no, flag):
		trainInd = {}
		x = {}
		t = {}
		mat = {}
		xTrain = {}
		tTrain = {}
		mat_train = {}

		if(flag == 0):
			x[no], t[no] = self.divide_track(self.track[no])
			mat[no] = self.track[no].groupby(["date", "キロ程"]).max()["高低左"].unstack().notnull().values
			trainInd[no] = int(len(self.track[no]) * self.trainPer)
		elif(flag == 1):
			x[no], t[no] = self.divide_equipment(self.equipment[no])
			mat[no] = self.track[no].groupby(["date", "キロ程"]).max()["高低左"].unstack().notnull().values
			trainInd[no] = int(len(self.equipment[no]) * self.trainPer)
		xTrain[no] = x[no][:trainInd[no]]
		tTrain[no] = t[no][:trainInd[no]]
		mat_train[no] = mat[no][:trainInd[no]]

		return xTrain[no], tTrain[no], mat_train[no]
	#------------------------------------

	#------------------------------------
	# testデータを読み込む
	def get_test_data(self, no):
		trainInd = {}
		x = {}
		t = {}
		mat = {}
		xTest = {}
		tTest = {}
		mat_test = {}

		if(flag == 0):
			x[no], t[no] = self.divide_track(self.track[no])
			mat[no] = self.track[no].groupby(["date", "キロ程"]).max()["高低左"].unstack().notnull().values
			trainInd[no] = int(len(self.track[no]) * self.trainPer)
		elif(flag == 1):
			x[no], t[no] = self.divide_equipment(self.equipment[no])
			mat[no] = self.track[no].groupby(["date", "キロ程"]).max()["高低左"].unstack().notnull().values
			trainInd[no] = int(len(self.equipment[no]) * self.trainPer)
		#trainInd[no] = int(len(self.track[no])*self.trainPer)
		xTest[no] = x[no][trainInd[no]:]
		tTest[no] = t[no][trainInd[no]:]
		mat_test[no] = mat[no][trainInd[no]:]

		return xTest[no], tTest[no], mat_test[no]
	#------------------------------------

	#------------------------------------
	def dump_file(self,filename,data):
		f = open(filename,'wb')
		pickle.dump(data,f)
		f.close
	#------------------------------------

	#------------------------------------
	def dump_data(self):
		fileind = ['A','B','C','D']

		for no in range(len(fileind)):
			fname_xTra = "xTrain_{}.binaryfile".format(fileind[no])
			# fname_xTes = "xTest_{}.binaryfile".format(fileind[no])
			fname_tTra = "tTrain_{}.binaryfile".format(fileind[no])
			# fname_tTes = "tTest_{}.binaryfile".format(fileind[no])

			self.dump_file(fname_xTra, self.train_xData[no])
			# self.dump_file(fname_xTes, self.test_xData[no])
			self.dump_file(fname_tTra, self.train_tData[no])
			# self.dump_file(fname_tTes, self.test_tData[no])
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
	mat = {}


	xTrain_e = {}
	tTrain_e = {}
	xTest_e = {}
	tTest_e = {}
	mat_e = {}


	myData = pre_processing()

	for no in ['A', 'B', 'C', 'D']:
		# trainデータについて
		xTrain[no], tTrain[no], mat[no]= myData.get_train_data(no,0)
		print("【xTrain_{}】\n{}\n".format(no, xTrain[no]))
		print("【tTrain_{}】\n{}\n".format(no, tTrain[no]))
		"""うまくいかない
		# ファイル出力
		# 説明変数
		fname = "xTrain_{}.txt".format(no)
		myData.file_output(fname, xTrain[no])
		# 目的変数
		fname = "tTrain_{}.txt".format(no)
		myData.file_output(fname, tTrain[no])"""

		# testデータについて
		xTest[no], tTest[no], mat[no] = myData.get_test_data(no,0)
		print("【xTest_{}】\n{}\n".format(no, xTest[no]))
		print("【tTest_{}】\n{}\n".format(no, tTest[no]))
		"""うまくいかない
		# ファイル出力
		# 説明変数
		fname = "xTest_{}.txt".format(no)
		myData.file_output(fname, xTest[no])
		# 目的変数
		fname = "tTest_{}.txt".format(no)
		myData.file_output(fname, tTest[no])"""

	for no in ['A','B','C','D']:
		#trainデータについて
		xTrain_e[no],tTrain[no], mat_e[no] = myData.get_train_data(no,1)
		print("【xTrain_e{}】\n{}\n".format(no, xTrain[no]))
		print("【tTrain_e{}】\n{}\n".format(no, xTrain[no]))

		xTest_e[no],tTrain_e[no],mat_e[no] = myData.get_test_data(no,1)
		print("【xTest_e{}】\n{}\n".format(no, xTest_e[no]))
		print("【tTest_e{}】\n{}\n".format(no, xTest_e[no]))
#メインの終わり
#-------------------
