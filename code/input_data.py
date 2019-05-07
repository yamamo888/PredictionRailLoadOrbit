# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import numpy

from six.moves import urllib
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
import prepare_data
import tarfile
import numpy as np
import pickle

SOURCE_URL = 'https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/'
data_path = "data"
real_filepath = 'TownCentre.pkl.gz'
cg_filepath = 'CGData.pkl.gz'

def maybe_download(filename,work_directory):
    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)

    filepath = os.path.join(work_directory,filename)

    if not tf.gfile.Exists(filepath):
        #filename = "TownCentreHeadImages.tar.bz2"
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        f = tarfile.open(filepath,"r:bz2")
        f.extractall(work_directory)
        print('Successfully downloaded and extracted', filename)

    return filepath


def dump_data(real_use,cg_use):
    if (not tf.gfile.Exists(os.path.join(data_path,real_filepath))) and real_use:
        prepare_data.prepare_dump(True,False)
    if (not tf.gfile.Exists(os.path.join(data_path,cg_filepath))) and cg_use:
        prepare_data.prepare_dump(False,True)

class Dataset():
    def __init__(self,filepath):
        if filepath == os.path.join(data_path,real_filepath):
            X,y,n = pickle.load(gzip.open(filepath, 'rb'))
            self.train , self.test = self.split(X,y,n)
        else:
            X,y = pickle.load(gzip.open(filepath, 'rb'))
            self.cgdata = data(X,y,n="cgdata")


    def split(self,X, y, n, split=0.9):
        itr, ite, trs, tes = [], [], set(), set()
        for i, name in enumerate(n):
            # Extract the person's ID.
            pid = int(name.split('_')[1])

            # Decide where to put that person.
            if pid in trs:
                itr.append(i)
            elif pid in tes:
                ite.append(i)
            else:
                if np.random.rand() < split:
                    itr.append(i)
                    trs.add(pid)
                else:
                    ite.append(i)
                    tes.add(pid)
        return data(X[itr], y[itr], [n[i] for i in itr]), data(X[ite], y[ite], [n[i] for i in ite])


class data():
    def __init__(self,X,y,n,dtype = tf.float32):
        images = X.astype(numpy.float32)
        self._images = numpy.multiply(images,1.0/255.0)
        self._labels = y
        self.n = n
        self._index_in_epoch = 0
        self._num_examples = X.shape[0]
        self._epochs_completed = 0

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
    		# Finished epoch
            self._epochs_completed += 1
    		# Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end],self._labels[start:end]


def read_data_sets(real_use = True,cg_use = False):
    maybe_download("TownCentreHeadImages.tar.bz2","data")

    dump_data(real_use,cg_use)

    real_data = Dataset(os.path.join(data_path,real_filepath))
    if cg_use and real_use:
        cg_data = Dataset(os.path.join(data_path,cg_filepath))
        return real_data, cg_data

    if real_use:
        return real_data

    if cg_use:
        return cg_data
