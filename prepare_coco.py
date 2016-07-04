# Import
import pdb
import json
import os
from sys import stdout
import scipy
import  cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import sys

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import pandas as pd
import nltk

# Setup
CWD = os.getcwd() + "/"
originalImagesPath     = CWD + '../data/coco/originalImages'
preprocessedImagesPath = CWD + '../data/coco/processedImages/'

caffe_root = '/Users/Grendel/caffe/'

vgg_ilsvrc_19_layoutFileName = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_ilsvrc_19_modelFileName  = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'

dataPath        = CWD + '../data/coco/'
annotation_path = dataPath + 'annotations/'
annotation_tr   = annotation_path + 'captions_train2014.json'
splitFileName   = dataPath + 'dataset_coco.json'

tr_data_path    = dataPath + 'train2014/'
te_data_path    = dataPath + 'test2014/'
val_data_path   = dataPath + 'val2014/'

experimentPrefix = '.exp1'

# Set up cafe and load the Net
sys.path.insert(0, caffe_root + 'python')
import caffe
from cnn_util import CNN
cnn = CNN(deploy=vgg_ilsvrc_19_layoutFileName,
          model=vgg_ilsvrc_19_modelFileName,
          batch_size=2,
          width=224,
          height=224)

# Load the training image & captions information
import json
with open(annotation_tr,'r') as f:
    stdout.write("Loading data from annotations...\n")
    caps_notes = json.load(f)

    stdout.write("Make dictionary...\n")
    corpus3 = ' '.join([note['caption'] for note in caps_notes['annotations']])
    words = nltk.FreqDist(corpus3.split()).most_common()
    wordsDict = {words[i][0]:i+2 for i in range(len(words))}
    with open(dataPath + 'dictionary.pkl', 'wb') as f:
        pickle.dump(wordsDict, f)

    stdout.write("Generating images data...\n")
    images_train     = []
    images_train_idx = {}
    for info in caps_notes['images']:
        images_train_idx[str(info['id'])] = len(images_train)
        images_train.append(tr_data_path + str(info['file_name']))

    stdout.write("Generating captions data...\n")
    cap_train = [(note['caption'], images_train_idx[str(note['image_id'])]) for note in caps_notes['annotations']]

    stdout.write("Finished generating data...\n")

stdout.write("Start extracting features...\n")
BATCH_SIZE = 10
for start, end in zip(range(0, len(images_train)+BATCH_SIZE, BATCH_SIZE), range(BATCH_SIZE, len(images_train)+BATCH_SIZE, BATCH_SIZE)):
    image_files = images_train[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])
    stdout.write("processing images %d to %d\n" % (start, end))
    stdout.flush()

print "Saving indices information: i -> image_id \n"
with open(dataPath + 'coco_align.train.indices.pkl', 'wb') as f:
    cPickle.dump(images_train_idx, f)

print "Saving features... \n"
with open(dataPath + 'coco_align.train.pkl', 'wb') as f:
    cPickle.dump(cap_train, f,-1)
    cPickle.dump(feat_flatten_list_train, f)




