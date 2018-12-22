<<<<<<< HEAD
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
	def __init__(self):
		# trainデータの割合
		self.trainPer = 0.8

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
		# 削除するNanのリスト
		list_del = np.array([], dtype=int)

		# 高低左のデータを基準とする
		nan_mat = self.shape_matrix(data, "高低左")

		for row in range(nan_mat.shape[0]):
			if(math.isnan(nan_mat[row][0]) == True):
					list_nan　=　np.append(list_nan,　i)
			elif(math.isnan(nan_mat[row][0]) == False):
				if(list_nan.shape[0] >= 10):
					list_del = np.append(list_del, list_nan)
					list_nan = np.array([], dtype=int)
				else:
					list_nan = np.array([], dtype=int)

		return list_del
	#------------------------------------

	#------------------------------------
	# 補完する欠損値のインデックスを取得
	def get_fill_index(self, mat):
		# NaNのインデックスリスト
		list_nan = np.array([], dtype=int)
		# 補完するNaNのリスト
		list_fill = np.zeros((1, 2),dtype=int)
		
		# 同じキロ程ごとに違うdateをみていく
		for col in range(mat.shape[1]):
			for row in range(mat.shape[0]):
				if(math.isnan(mat[row][col]) == True):
					list_nan = np.append(list_nan, [[row, col]], axis=0)
				elif(math.isnan(mat[row][col]) == False):
					if(0 < list_nan.shape[0] < 10):
						list_fill = np.append(list_fill, list_nan, axis=0)
						list_nan = np.zeros((1, 2), dtype=int)
					else:
						list_nan = np.zeros((1, 2), dtype=int)

		return list_fill[1:,:]
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
		rnd = np.random.randint(ave - std, ave + std)
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
		data_mat = mat[row-4:row+5][col-4:col+5]

		# 欠損値の予測
		value = np.nansum(gauss * data_mat)
		# 補完する
		mat[row][col] = value

		return mat
	#------------------------------------

	#------------------------------------
	# インデックスで場合分けをして補完
	def complement(self, mat):
		# [date x キロ程]の行列を取得
		#newMat = self.shape_matrix(data, target)

		# 行、列のサイズを取得
		row_max = mat.shape[0]
		col_max = mat.shape[1]

		# 補完するインデックスを取得
		fill = self.get_fill_index(mat)

		# 補完
		for i in range(fill.shape[0]):
			row, col = fill[i]
			# 端の欠損値は平均で補完
			if(row < 4 | (row_max - row) < 4 | col < 4 | (col_max - col) < 4):
				mat = self.ave_complement(mat, row, col)
			# 中の欠損値はガウシアンで補完
			else:
				mat = self.gauss_complement(mat, row, col)

		return mat
	#------------------------------------

	#------------------------------------
	# 欠損値に対する処理を行う
	def missing_values(self, data):
		# 積み木の形にする
		newData = self.(data)

		# 削除するインデックスを取得
		delete = self.get_del_index(data)
		# 削除ターン
		for i in range(delete.shape[0]):
			newData = np.delete(newData, delete[i])

		# 補完ターン
		for i range(self.track_label.shape[0]):
			# 積み木のi番目のスライスをもってくる
			newMat = newData[i]
			# 補完
			newMat = self.complement(newMat)
			# 補完済みのスライスを積み木に戻す
			newData[i] = newMat

		# 積み木の形で返す
		return newData
	#------------------------------------

	#------------------------------------
	# 説明変数と目的変数に分ける
	def divide_track(self, data):
		newData = self.missing_values(data)

		# 目的変数は高低左
		x = np.delete(newData, 0, axis=0)
		t = newData[0,:,:]

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
			trainInd[no] = int(len(self.track[no]) * self.trainPer)
		elif(flag == 1):
			x[no], t[no] = self.divide_equipment(self.equipment[no])
			trainInd[no] = int(len(self.equipment[no]) * self.trainPer)
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
			trainInd[no] = int(len(self.equipment[no]) * self.trainPer)
		#trainInd[no] = int(len(self.track[no])*self.trainPer)
		xTest[no] = x[no][trainInd[no]:]
		tTest[no] = t[no][trainInd[no]:]


		return xTest[no], tTest[no]
	#------------------------------------
"""
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
"""
# pre_processingクラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	xTrain= {}
	tTrain = {}
	xTest = {}
	tTest = {}

	xTrain_e = {}
	tTrain_e = {}
	xTest_e = {}
	tTest_e = {}

	myData = pre_processing()

	for no in ['A', 'B', 'C', 'D']:
		# trainデータについて
		xTrain[no], tTrain[no]= myData.get_train_data(no, 0)
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
		xTest[no], tTest[no] = myData.get_test_data(no, 0)
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
		xTrain_e[no],tTrain[no] = myData.get_train_data(no, 1)
		print("【xTrain_e{}】\n{}\n".format(no, xTrain[no]))
		print("【tTrain_e{}】\n{}\n".format(no, xTrain[no]))

		xTest_e[no],tTrain_e[no] = myData.get_test_data(no, 1)
		print("【xTest_e{}】\n{}\n".format(no, xTest_e[no]))
		print("【tTest_e{}】\n{}\n".format(no, xTest_e[no]))
#メインの終わり
#-------------------
=======
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
	dataPath = '../data' # データが格納させているフォルダ名

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
	# 欠損値を予測する
	# ガウシアンカーネルと協調フィルタリングの考え方を用いる
	# 未実装
	def gauss_complement(self, data, target, index):
		newData = self.shape_matrix(data, target)
		row, col = index
		# ガウシアンカーネルを作成
		gauss = self.gaussian_kernel(9)

		# ガウシアンカーネルと同じshapeで中心が欠損値の行列を作成
		# まずは全要素Nanにする
		mat = np.zeros_like(gauss)
		mat[:,:] = np.nan
		# newDataから値をもってくる
		mat = newData[row][col]

		# 欠損値の予測
		value = np.nansum(gauss * mat)

		return value
	#------------------------------------

	#------------------------------------
	# 削除する欠損値のインデックスを取得する
	def get_del_index(self, data):
		# NaNのインデックスリスト
		list_nan = np.array([], dtype=int)
		# 削除するNanのリスト
		list_del = np.array([], dtype=int)

		# 高低左のデータを基準とする
		nan_mat = self.shape_matrix(data, "高低左")
		for row in range(nan_mat.shape[0]):
			if(math.isnan(nan_mat[row][0]) == True):
				list_nan = np.append(list_nan,row)
			elif(math.isnan(nan_mat[row][0]) == False):
				if(list_nan.shape[0] >= 10):
					list_del = np.append(list_del, list_nan)
					list_nan = np.array([], dtype=int)
				else:
					list_nan = np.array([], dtype=int)

		return list_del
	#------------------------------------

	#------------------------------------
	# 補完する欠損値のインデックスを取得する
	# 引数であるdataは不要なデータを削除済みのデータセットであることが理想
	def get_fill_index(self, data):
		# NaNのインデックスリスト
		list_nan = np.array([], dtype=int)
		# 補完するNanのリスト
		list_fill = np.zeros((1, 2), dtype=int)

		# 高低左のデータを基準とする
		nan_mat = self.shape_matrix(data, "高低左")
		# 同じキロ程でみていく
		for col in range(nan_mat.shape[1]):
			for row in range(nan_mat.shape[0]):
				if(math.isnan(nan_mat[row][col]) == True):
					list_nan = np.append(list_nan, [[row, col]], axis=0)
				elif(math.isnan(nan_mat[row][col]) == False):
					if(0 < list_nan.shape[0] < 10):
						list_fill = np.append(list_fill, list_nan, axis=0)
					else:
						list_nan = np.zeros((1, 2), dtype=int)

		return list_fill[1:,:]
	#------------------------------------

	#------------------------------------
	# 欠損値に対する処理を行う
	def missing_values(self, data):
		newData = data
		
		delete = self.get_del_index(data)
		
		
		##-----------------------------
		##--------------追加-----------
		reshaped_data = self.data_shape(data)
		##-----------------------------
		##-----------------------------

		# 削除
		for i in range(len(delete)):
			newData = np.delete(newData, delete[i])
		#↓は未実装
		# 補完
		#for j in range(len(fill)):
			#newData = self.gauss_complement(newData, fill[i])

		#return newData
	#------------------------------------

	#------------------------------------
	# データの型調整
	def data_shape(self,data):
		new_data = np.array([])
		data = data.values
		pdb.set_trace()	
		for i in range(data.shape[1]):
			for j in range(data.T[i+2].shape[0]):
				if j==0:
					new_data = np.append(new_data,data.T[i+2][j])
				elif data.T[1][j] < data.T[1][j-1]:
					pdb.set_trace()	
		reshaped_data = 0
		return reshaped_data

	#------------------------------------
	# 説明変数と目的変数に分ける
	# まだこれを使える状態にはなっていない
	def divide_track(self, data):
		mat = self.missing_values(data)

		# 目的変数は高低左
		x = np.delete(mat, 0, axis=0)
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
			trainInd[no] = int(len(self.track[no]) * self.trainPer)
		elif(flag == 1):
			x[no], t[no] = self.divide_equipment(self.equipment[no])
			trainInd[no] = int(len(self.equipment[no]) * self.trainPer)
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
			trainInd[no] = int(len(self.equipment[no]) * self.trainPer)
		#trainInd[no] = int(len(self.track[no])*self.trainPer)
		xTest[no] = x[no][trainInd[no]:]
		tTest[no] = t[no][trainInd[no]:]


		return xTest[no], tTest[no]
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

	xTrain_e = {}
	tTrain_e = {}
	xTest_e = {}
	tTest_e = {}

	myData = pre_processing()

	for no in ['A', 'B', 'C', 'D']:
		# trainデータについて
		xTrain[no], tTrain[no]= myData.get_train_data(no, 0)
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
		xTest[no], tTest[no] = myData.get_test_data(no, 0)
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
		xTrain_e[no],tTrain[no] = myData.get_train_data(no, 1)
		print("【xTrain_e{}】\n{}\n".format(no, xTrain[no]))
		print("【tTrain_e{}】\n{}\n".format(no, xTrain[no]))

		xTest_e[no],tTrain_e[no] = myData.get_test_data(no, 1)
		print("【xTest_e{}】\n{}\n".format(no, xTest_e[no]))
		print("【tTest_e{}】\n{}\n".format(no, xTest_e[no]))
#メインの終わり
#-------------------
>>>>>>> a3f595782894fee333a279776bb04e7bf94d66e2
