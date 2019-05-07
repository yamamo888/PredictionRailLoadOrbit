import os
import sys
import re
import inspect
import json
import pickle
import gzip
from os.path import join as pjoin
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
import h5py
import pdb

def load_towncentre(datadir, normalize_angles=True):
    panre = re.compile('pan = ([+-]?\d+\.\d+)\n')
    valre = re.compile('valid = ([01])\n')
    angles = []
    images = []
    names = []
    for father in os.listdir(datadir):
      try:
        for son in os.listdir(pjoin(datadir, father)):
            if not son.endswith('.txt'):
                continue

            lpan, lval = open(pjoin(datadir, father, son)).readlines()
            if int(valre.match(lval).group(1)) == 0:
                continue

            angles.append(float(panre.match(lpan).group(1)))
            # Now search for the corresponding filename, unfortunately, it has more numbers encoded...
            fnames = [f for f in os.listdir(pjoin(datadir, father)) if f.startswith(son.split('.')[0]) and not f.endswith('.txt')]
            assert len(fnames) == 1, "lolwut"
            names.append(fnames[0])
            images.append(cv2.imread(pjoin(datadir, father, fnames[0]), flags=cv2.IMREAD_COLOR))
      except NotADirectoryError:
        pass

    if normalize_angles:
        angles = [(a + 360*2) % 360 for a in angles]

    return images, angles, names

def load_cgdata(datadir):
    images = []
    angles = []
    try:
        for son in os.listdir(datadir):
            fname = son.split('.')[0]
            angles.append(float(fname.split('_')[1]))
            images.append(cv2.imread(pjoin(datadir, son), flags=cv2.IMREAD_COLOR))
        images = np.array(images)
        angles = np.array(angles)

    except NotADirectoryError:
        pass

    return images, angles

def flipany(a, dim):
    """
    `flipany(a, 0)` is equivalent to `flipud(a)`,
    `flipany(a, 1)` is equivalent to `fliplr(a)` and the rest follows naturally.
    """
    # Put the axis in front, flip that axis, then move it back.
    return np.swapaxes(np.swapaxes(a, 0, dim)[::-1], 0, dim)

def flip_images(images):
    return [flipany(img,dim=1) for img in images]

def flip_angles(angles):
    return [360 - ang for ang in angles]

def scale_all(images, size=(50, 50)):
    return [cv2.resize(im, size, interpolation=cv2.INTER_LANCZOS4) for im in images]

def prepare_dump(real_use = True,cg_use = False):
    data_path = "data"
    towncentre_path = pjoin(data_path,"TownCentreHeadImages")
    cgdata_path = pjoin(data_path,"cg_data")
    if real_use:
        img, angle, name = load_towncentre(towncentre_path)
        img = scale_all(img,(50,50))
        x = np.array(img + flip_images(img))
        y = np.array(angle + flip_angles(angle))
        n = name + name
        f = gzip.open(pjoin(data_path,'TownCentre.pkl.gz'),'wb+')
        pickle.dump((x,y,n),f)
    if cg_use:
        cg_img, cg_angle = load_cgdata(cgdata_path)
        f = gzip.open(pjoin(data_path,'CGData.pkl.gz'),'wb+')
        pickle.dump((cg_img,cg_angle),f)
