# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cluster import KMeans
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib.pylab as plt
import sys

#===========================
# レイヤーの関数
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

# 1D convolution layer
def conv1d_relu(inputs, w, b, stride):
	# tf.nn.conv1d(input,filter,strides,padding)
	#filter: [kernel, output_depth, input_depth]
	# padding='SAME' はゼロパティングしている
	conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 1D deconvolution
def conv1d_t_relu(inputs, w, b, output_shape, stride):
	conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D convolution
def conv2d_relu(inputs, w, b, stride):
	# tf.nn.conv2d(input,filter,strides,padding)
	# filter: [kernel, output_depth, input_depth]
	# input 4次元([batch, in_height, in_width, in_channels])のテンソルを渡す
	# filter 畳込みでinputテンソルとの積和に使用するweightにあたる
	# stride （=１画素ずつではなく、数画素ずつフィルタの適用範囲を計算するための値)を指定
	# ただし指定は[1, stride, stride, 1]と先頭と最後は１固定とする
	conv = tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D deconvolution layer
def conv2d_t_sigmoid(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.sigmoid(conv)
	return conv

# 2D deconvolution layer
def conv2d_t_relu(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D deconvolution layer
def conv2d_t(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	return conv

# fc layer with ReLU
def fc_relu(inputs, w, b, keepProb=1.0):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	fc = tf.nn.relu(fc)
	return fc

# fc layer
def fc(inputs, w, b, keepProb=1.0):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	return fc

# fc layer with softmax
def fc_sigmoid(inputs, w, b, keepProb=1.0):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	fc = tf.nn.sigmoid(fc)
	return fc
#===========================

#===========================
# エンコーダ
# 画像をz_dim次元のベクトルにエンコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def encoderR(x, z_dim, reuse=False, keepProb = 1.0):
	with tf.variable_scope('encoderR') as scope:
		if reuse:
			scope.reuse_variables()

		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 50/2 = 25
		convW1 = weight_variable("convW1", [3, 3, 3, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])

		# 25/2 = 13
		convW2 = weight_variable("convW2", [3, 3, 32, 64])
		convB2 = bias_variable("convB2", [64])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])

		#--------------
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])

		# 13 x 13 x 32 -> z-dim
		fcW1 = weight_variable("fcW1", [conv2size, z_dim])
		fcB1 = bias_variable("fcB1", [z_dim])
		#fc1 = fc_relu(conv2, fcW1, fcB1, keepProb)
		fc1 = fc(conv2, fcW1, fcB1, keepProb)
		#--------------

		return fc1
#===========================

#===========================
# デコーダ
# z_dim次元の画像にデコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def decoderR(z,z_dim,reuse=False, keepProb = 1.0):
	with tf.variable_scope('decoderR') as scope:
		if reuse:
			scope.reuse_variables()

		#--------------
		# embeddingベクトルを特徴マップに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		fcW1 = weight_variable("fcW1", [z_dim, 13*13*64])
		fcB1 = bias_variable("fcB1", [13*13*64])
		fc1 = fc_relu(z, fcW1, fcB1, keepProb)

		batchSize = tf.shape(fc1)[0]
		fc1 = tf.reshape(fc1, tf.stack([batchSize, 13, 13, 64]))
		#--------------

		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 7 x 2 = 14
		convW1 = weight_variable("convW1", [3, 3, 32, 64])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_t_relu(fc1, convW1, convB1, output_shape=[batchSize,25,25,32], stride=[1,2,2,1])

		# 14 x 2 = 28
		convW2 = weight_variable("convW2", [3, 3, 3, 32])
		convB2 = bias_variable("convB2", [3])
		output = conv2d_t_relu(conv1, convW2, convB2, output_shape=[batchSize,50,50,3], stride=[1,2,2,1])
		#output = conv2d_t_sigmoid(conv1, convW2, convB2, output_shape=[batchSize,28,28,1], stride=[1,2,2,1])

		return output
#===========================

#===========================
# D Network
#
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def DNet(x, z_dim=1, reuse=False, keepProb=1.0):
	with tf.variable_scope('DNet') as scope:
		if reuse:
			scope.reuse_variables()

		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 28/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 3, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])

		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, 32, 32])
		convB2 = bias_variable("convB2", [32])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])

		#--------------
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])

		# 7 x 7 x 32 -> z-dim
		fcW1 = weight_variable("fcW1", [conv2size, z_dim])
		fcB1 = bias_variable("fcB1", [z_dim])
		fc1 = fc_sigmoid(conv2, fcW1, fcB1, keepProb)
		#--------------

		return fc1
#===========================
def plotImg(x,y,path):
	#--------------
	# 画像を保存
	plt.close()

	fig, figInds = plt.subplots(nrows=2, ncols=x.shape[0], sharex=True)

	for figInd in np.arange(x.shape[0]):
		fig0 = figInds[0][figInd].imshow(x[figInd,:,:,0],cmap="gray")
		fig1 = figInds[1][figInd].imshow(y[figInd,:,:,0],cmap="gray")

		# ticks, axisを隠す
		fig0.axes.get_xaxis().set_visible(False)
		fig0.axes.get_yaxis().set_visible(False)
		fig0.axes.get_xaxis().set_ticks([])
		fig0.axes.get_yaxis().set_ticks([])
		fig1.axes.get_xaxis().set_visible(False)
		fig1.axes.get_yaxis().set_visible(False)
		fig1.axes.get_xaxis().set_ticks([])
		fig1.axes.get_yaxis().set_ticks([])

	plt.savefig(path)
	#--------------
def mmnorm(x):
	max = 255
	min = 0
	newx = (x-min)/(max-min)
	return newx
def norm(x):
	max = 255
	min = 0
	newx = x
	newx[newx > max] = max
	newx[newx < min] = min
	return newx
#=========================


if __name__ == "__main__":

    # Rの二乗誤差の重み係数
    lambdaR = 0.4

    # log(0)と0割防止用
    lambdaSmall = 10e-8

    # 予測結果に対する閾値
    threFake = 0.5

    # Rの誤差の閾値
    threLossR = 50

    # Dの誤差の閾値
    threLossD = -10e-8

    # バッチデータ数
    batchSize = 300

    # プロットする画像数
    nPlotImg = 10

    isStop = False
    isEmbedSampling = True
    isTrain = True
    isVisualize = True

    noiseSigma = 5
    z_dim_R =10
    nIte = 5000

    #============================
    #データの読み込み
    # pdb.set_trace()
    realImg = input_data.read_data_sets()
    TrainLabels = realImg.train.labels
    TrainData = realImg.train.images
    batchNum = len(TrainLabels)//batchSize
    TestLabels = realImg.test.labels
    TestData = realImg.test.images
    #============================


    #============================
    #gan
    xTrue = tf.placeholder(tf.float32, shape=[None, 50, 50, 3])
    xFake = tf.placeholder(tf.float32, shape=[None, 50, 50, 3])
    xTest = tf.placeholder(tf.float32, shape=[None, 50, 50, 3])
    xTestNoise = tf.placeholder(tf.float32, shape=[None, 50, 50, 3])

    # 学習用
    encoderR_train = encoderR(xFake, z_dim_R, keepProb=1.0)
    decoderR_train = decoderR(encoderR_train, z_dim_R, keepProb=1.0)

    # テスト用
    encoderR_test = encoderR(xTestNoise, z_dim_R, reuse=True, keepProb=1.0)
    decoderR_test = decoderR(encoderR_test, z_dim_R, reuse=True, keepProb=1.0)
    #============================


    #===========================
    # 損失関数の設定

    #学習用
    predictFake_train = DNet(decoderR_train, keepProb=1.0)
    predictTrue_train = DNet(xTrue, reuse=True, keepProb=1.0)

    lossR = tf.reduce_mean(tf.square(decoderR_train - xTrue))
    lossRAll = tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + lambdaR * lossR
    lossD = tf.reduce_mean(tf.log(predictTrue_train  + lambdaSmall)) + tf.reduce_mean(tf.log(1 - predictFake_train +  lambdaSmall))

    # R & Dの変数
    Rvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoderR") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoderR")
    Dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DNet")

    #=============================

    #=============================
    #trainer_op
    trainerR = tf.train.AdamOptimizer(1e-3).minimize(lossR, var_list=Rvars)
    trainerRAll = tf.train.AdamOptimizer(1e-3).minimize(lossRAll, var_list=Rvars)
    trainerD = tf.train.AdamOptimizer(1e-3).minimize(-lossD, var_list=Dvars) # 0に近づけたい

    #=============================

    # メイン
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    lossR_values = []
    lossRAll_values = []
    lossD_values = []

    batchInd = 0
    ite = 0

    while not isStop:

        ite = ite + 1
    	#--------------
    	# 学習データの作成
        if batchInd == batchNum-1:
            batchInd = 0
        batch = TrainData[batchInd*batchSize:(batchInd+1)*batchSize]
        batch_x = np.reshape(batch,(batchSize,50,50,3))*255
        batchInd += 1
        # ノイズを追加する(ガウシアンノイズ)
        # 正規分布に従う乱数を出力
        batch_x_fake = batch_x + np.random.normal(0,noiseSigma,batch_x.shape)
        batch_x_fake = norm(batch_x_fake)

        #--------------
    	#==============
        # 学習
        if isTrain:
    		# training D network with batch_x & batch_x_fake
            _, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run([trainerD, lossD, predictFake_train, predictTrue_train],feed_dict={xTrue: batch_x,xFake: batch_x_fake})

            # training R network with batch_x & batch_x_fake
            _, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run([trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],feed_dict={xTrue: batch_x, xFake: batch_x_fake})

            # Re-training R network with batch_x & batch_x_fake
            _, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run([trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],feed_dict={xTrue: batch_x, xFake: batch_x_fake})
        # もし誤差が下がらない場合は終了
        if (ite > 2000) & (lossD_value < -10):
            isTrain = False
        if ite >= nIte:
            isStop = True
        #==============
        # 損失の記録
        lossR_values.append(lossR_value)
        lossRAll_values.append(lossRAll_value)
        lossD_values.append(lossD_value)
        if (ite %100 == 0):
            print("#%d , lossR=%f, lossRAll=%f, lossD=%f" % (ite,lossR_value,lossRAll_value,lossD_value))

        #=======================
        #test
        if (ite%1000 == 0):
            # if isVisualize:
            plt.close()
            test_x = np.reshape(TestData,(len(TestData),50,50,3))*255
            test_x_noise = norm(test_x + np.random.normal(0,noiseSigma,test_x.shape))
            decoderR_test_value = sess.run(decoderR_test,feed_dict={xTestNoise:test_x_noise})

            fig, figInds = plt.subplots(nrows=3, ncols=nPlotImg)
            pdb.set_trace()
            for i in np.arange(nPlotImg):
                figInds[0,i].imshow(mmnorm(test_x[i]))
                figInds[1,i].imshow(mmnorm(test_x_noise[i]))
                figInds[2,i].imshow(mmnorm(decoderR_test_value[i]))
                # ticks, axisを隠す
                figInds[0,i].axis("off")
                figInds[1,i].axis("off")
                figInds[2,i].axis("off")

            path = os.path.join("visualization","img_test_{}_{}.png".format(z_dim_R,ite))
            plt.savefig(path)
        #========================
