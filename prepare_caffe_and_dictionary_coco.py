# Setup 

originalImagesPath = 'data/coco/originalImages'
preprocessedImagesPath = 'data/coco/processedImages/'

caffe_root = '/home/intuinno/codegit/caffe/'

vgg_ilsvrc_19_layoutFileName = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_ilsvrc_19_modelFileName = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'

dataPath = 'data/coco/'
annotation_path = dataPath + 'annotations/captions_train2014.json'
splitFileName = dataPath + 'dataset_coco.json'

experimentPrefix = '.exp1'


# Import

import pdb
from sys import stdout
import scipy
import  cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os

import pandas as pd
import nltk

caffe.set_device(1)
caffe.set_mode_gpu()

net = caffe.Net(vgg_ilsvrc_19_layoutFileName,
                vgg_ilsvrc_19_modelFileName,
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Create file list 
# coco.devImages.txt 
# coco.trainImages.txt 
# coco.testImages.txt

import json
from pprint import pprint


with open(splitFileName) as f:
    data = json.load(f)

df = pd.DataFrame(data['images'])

files = [ 'dev','test','train']

dataDict = {}

dataDict['dev'] = df[df.split == 'val']
dataDict['test'] = df[df.split == 'test']
dataDict['restval'] = df[df.split == 'restval']
dataDict['train'] = df[df.split == 'train']

for f in files:
    dataDict[f]['filename'].to_csv(dataPath + 'coco.' + f + 'Images.txt',index=False)
    

def buildCapDict(sentences):
    return [s[u'raw'] for s in sentences ]

df['captions'] = df.apply(lambda row: buildCapDict(row['sentences']), axis=1)

capDict = df.loc[:,['filename', 'captions']].set_index('filename').to_dict()

capDict = capDict['captions']

# Let's build dictionary

# Let's make dictionary

corpus = df['captions'].values
corpus2 = [' '.join(c) for c in corpus]
corpus3 = ' '.join(corpus2)

words = nltk.FreqDist(corpus3.split()).most_common()

wordsDict = {words[i][0]:i+2 for i in range(len(words))}

with open(dataPath + 'dictionary.pkl', 'wb') as f:
    pickle.dump(wordsDict, f)