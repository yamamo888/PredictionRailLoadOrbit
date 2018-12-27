import datetime as dt
import pandas as pd
import numpy as np
import math
import os
#import matplotlib.pylab as plt   #折れ線グラフを作るやつ
import matplotlib.pyplot as plt
import pickle
import pdb


#-------------------
# pre_processingクラスの定義定義始まり
class pre_processing:
	dataPath = '../data' # データが格納されているフォルダ名

	#------------------------------------
	def __init__(self):
		# testデータの割合
		self.testPer = 0.2

		# dateとキロ程を除いたtrackデータのラベル
		self.track_label = np.array(["高低左", "高低右", "通り左", "通り右", "水準", "軌間", "速度"])

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
	# [date x キロ程]で中身がtargetの行列を作成
	def shape_matrix(self, data, target):
		newMat = data.groupby(["date", "キロ程"]).max()[target].unstack().values

		return newMat
	#------------------------------------

	#------------------------------------
	# ガウシアンカーネルを作成
	def gaussian_kernel(self, size):
		if size % 2 == 0:
			print('kernel size should be odd')
			return

		sigma = (size - 1) / 2

		# [0,size]を[-sigma,sigma]にずらす
		x = y = np.arange(0, size) - sigma
		X, Y = np.meshgrid(x, y)

		gauss = np.exp(-(np.square(X) + np.square(Y)) / (2 * np.square(sigma)) / (2 * np.pi * np.square(sigma)))

		# 総和が1になるように正規化
		kernel = gauss / np.sum(gauss)

		return kernel
	#------------------------------------

	#------------------------------------
	# 削除する欠損値のインデックスを取得
	def get_del_index(self, data):
		# NaNのインデックスリスト
		list_nan = np.array([], dtype=int)
		# 削除するNaNのリスト
		list_del = np.array([], dtype=int)

		# 高低左のデータを基準とする
		nan_mat = self.shape_matrix(data, "高低左")

		for row in range(nan_mat.shape[0]):
			if(math.isnan(nan_mat[row][0]) == True):
				list_nan = np.append(list_nan, row)
			elif(math.isnan(nan_mat[row][0]) == False):
				if(list_nan.shape[0] >= 10):
					list_del = np.append(list_del, list_nan)
					list_nan = np.array([], dtype=int)
				else:
					list_nan = np.array([], dtype=int)

		return list_del
	#------------------------------------

	#------------------------------------
	# 欠損値を補完
	# 平均からばらつきを考慮して補完
	def ave_complement(self, mat, row, col):
		# キロ程を1列分取り出す
		kilo = mat[:,col]
		# 平均値
		ave = np.nanmean(kilo)
		# 標準偏差
		std = np.nanstd(kilo)
		# 正規分布に従うとし標準偏差の範囲内でランダムに数字を作成
		rnd = (ave - std) * np.random.rand() + ave + std
		# 補完
		mat[row][col] = rnd

		return mat
	#------------------------------------

	#------------------------------------
	# 欠損値を予測
	# ガウシアンカーネルと協調フィルタリングの考え方を用いる
	def gauss_complement(self, mat, row, col):
		# ガウシアンカーネルを作成
		gauss = self.gaussian_kernel(9)

		# ガウシアンカーネルと同じshapeで中心が欠損値の行列を作成
		# 全要素をNaNにする
		data_mat = np.full_like(gauss, np.nan)
		# matから値をもってくる
		data_mat = mat[row-4:row+5, col-4:col+5]
		# 欠損値の予測
		value = np.nansum(gauss * data_mat)
		# 補完する
		mat[row][col] = value

		return mat
	#------------------------------------

	#------------------------------------
	# インデックスで場合分けをして補完
	def complement(self, mat):
		# object型からfloat型にキャスト
		newMat = mat.astype(float)

		# 行、列のサイズを取得
		row_max = newMat.shape[0]
		col_max = newMat.shape[1]
		#pdb.set_trace()
		# 補完するインデックスを取得
		row, col = np.where(np.isnan(newMat) == True)

		# 補完
		for i in range(row.shape[0]):
			if(row[i] < 4 or (row_max - row[i]) <= 4 or col[i] < 4 or (col_max - col[i]) <= 4):
				# 端の欠損値は平均で補完
				newMat = self.ave_complement(newMat, row[i], col[i])
			else:
				# 中の欠損値はガウシアンで補完
				newMat = self.gauss_complement(newMat, row[i], col[i])

		return newMat
	#------------------------------------

	#------------------------------------
	# 欠損値に対する処理を行う
	def missing_values(self, data):
		# 積み木の形にする
		#newData = self.data_reshape(data)
	# 削除するインデックスを取得
		delete = self.get_del_index(data)
		# 反転
		delete = delete[::-1]
		print("start reshape")
		#pdb.set_trace()
		#pandas をnumpyに
		numpy_data = data.values
		
		data_kiro = np.max(numpy_data.T[1])-np.min(numpy_data.T[1])+1			

		newData = np.reshape(numpy_data.T[2], (int(numpy_data.shape[0]/data_kiro),data_kiro))
		print("success reshape!!")

		# 削除ターン
		print("start delete")
		for i in range(delete.shape[0]):
			newData = np.delete(newData, delete[i], 0)
		print("success delete!!")

		# 補完ターン
		print("start complement")
		# とりあえず今は高低左についてのみ処理をする
		newMat = newData
		newMat = self.complement(newMat)
		newData = newMat
		"""
		for j in range(self.track_label.shape[0]):
			# 積み木のi番目のスライスをもってくる
			newMat = newData[j,:,:]
			# 補完
			newMat = self.complement(newMat)
			# 補完済みのスライスを積み木に戻す
			newData[j,:,:] = newMat
		"""
		print("success complement!!")
		#pdb.set_trace()
		return newData
	#------------------------------------

	#------------------------------------
	# データの型調整
	def data_reshape(self,data):

		reshaped_data = []

		for i in range(data.shape[1]-2):
			data_new = np.reshape(data.values.T[i+2],(365,27906))
			reshaped_data.append(data_new)
			#print("",i)
		#pdb.set_trace()
		numpy_data = np.array(reshaped_data)

		return numpy_data
	#------------------------------------

	#------------------------------------
	# 説明変数と目的変数に分ける
	def divide_track(self, data):
		newData = self.missing_values(data)

		# 目的変数は高低左
		return newData
		"""
		あとで拡張する
		x = np.delete(newData, 0, axis=0)
		t = newData[0,:,:]

		return x, t
		"""
	#------------------------------------

	#------------------------------------
	def divide_equipment(self, data):
		x = data.drop(["橋りょう"], axis=1).values
		t = data[["橋りょう"]].values

		return x, t
	#------------------------------------

	#------------------------------------
	# trainデータとtestデータに分けて取得
	def get_divide_data(self, no, flag):
		# flagでtrackかequipmentを分ける
		if(flag == 0):
			x = self.divide_track(self.track[no])
			#あとで拡張する
			#x, t = self.divide_track(self.track[no])
			testInd = int(len(self.track[no]) * self.testPer)
		elif(flag == 1):
			x = self.divide_equipment(self.equipment[no])
			#あとで拡張する
			#x, t = self.divide_equipment(self.equipment[no])
			testInd = int(len(self.equipment[no]) * self.testPer)

		# trainデータはすべてのデータ
		xTrain = x
		#tTrain = t
		# testデータははじめの2割
		xTest = x[:testInd]
		#tTest = t[:testInd]
		
		return xTrain, xTest
		#return xTrain, tTrain, xTest, tTest
	#------------------------------------

	#------------------------------------
	# pickleでdumpする
	def dump_file(self, filename, data):
		# ファイルの出力先
		fullpath = os.path.join(self.dataPath, filename)

		f = open(fullpath, 'wb')
		pickle.dump(data, f)
		f.close
	#------------------------------------

	#------------------------------------
	# 前処理後のデータをバイナリファイルとして出力
	def dump_data(self, no, flag):
		# trainデータ、testデータを読み込む
		xTrain, xTest = self.get_divide_data(no, flag)
		#あとで拡張する
		#xTrain, tTrain, xTest, tTest = self.get_divide_data(no, flag)

		# flagでtrackかequipmentを分ける
		if(flag == 0):
			# 名前付け
			fname_xTrain = "track_xTrain_{}.binaryfile".format(no)
			#fname_tTrain = "track_tTrain_{}.binaryfile".format(no)
			fname_xTest = "track_xTest_{}.binaryfile".format(no)
			#fname_tTest = "track_tTest_{}.binaryfile".format(no)
			# 出力
			self.dump_file(fname_xTrain, xTrain)
			#self.dump_file(fname_tTrain, tTrain)
			self.dump_file(fname_xTest, xTest)
			#self.dump_file(fname_tTest, tTest)
		elif(flag == 1):
			# 名前付け
			fname_xTrain = "equipment_xTrain_{}.binaryfile".format(no)
			#fname_tTrain = "equipment_tTrain_{}.binaryfile".format(no)
			fname_xTest = "equipment_xTest_{}.binaryfile".format(no)
			#fname_tTest = "equipment_tTest_{}.binaryfile".format(no)
			# 出力
			self.dump_file(fname_xTrain, xTrain)
			#self.dump_file(fname_tTrain, tTrain)
			self.dump_file(fname_xTest, xTest)
			#self.dump_file(fname_tTest, tTest)
	#------------------------------------
# pre_processingクラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# pre_processingクラスの呼び出し
	myData = pre_processing()

	for no in ['A', 'B', 'C', 'D']:
		# trackについて
		myData.dump_data(no, 0)
		# equipmentについて
		myData.dump_data(no, 1)
#メインの終わり
#-------------------
