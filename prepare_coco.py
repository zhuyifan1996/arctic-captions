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

import pandas as pd
import nltk

# Setup 
CWD = os.getcwd() + "/"
originalImagesPath     = CWD + '../../data/coco/originalImages'
preprocessedImagesPath = CWD + '../../data/coco/processedImages/'

caffe_root = '/Users/Grendel/caffe/'

vgg_ilsvrc_19_layoutFileName = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_ilsvrc_19_modelFileName  = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'

dataPath        = CWD + '../../data/coco/'
annotation_path = dataPath + 'annotations/'
annotation_tr   = annotation_path + 'captions_train2014.json'
splitFileName   = dataPath + 'dataset_coco.json'

tr_data_path    = dataPath + 'train2014/'
te_data_path    = dataPath + 'test2014/'
val_data_path   = dataPath + 'val2014/'

experimentPrefix = '.exp1'

# Set up cafe and load the Net
caffe.set_device(0)
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

# Make the dictionary form the captions of the training data
import json
with open(annotation_tr,'r') as f:
    caps_notes = json.load(f)
    corpus3 = ' '.join([note['caption'] for note in caps_notes['annotations']])
words = nltk.FreqDist(corpus3.split()).most_common()

wordsDict = {words[i][0]:i+2 for i in range(len(words))}

with open(dataPath + 'dictionary.pkl', 'wb') as f:
    pickle.dump(wordsDict, f)

files = [ 'val','test','train']

for fname in files:
    print fname 
    f = open(dataPath + "annotations/" + 'coco.' + fname + 'Images.txt')
    counter = 0
    
    imageList = [i for i in f]
    numImage = len(imageList)
    
    result = np.empty((numImage, 100352))

    for i in range(numImage):
        fn = imageList[i].rstrip()
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image( dataPath + fname + "2014/" +  fn))
        out = net.forward()
        feat = net.blobs['conv5_4'].data[0]
        reshapeFeat = np.swapaxes(feat, 0,2)
        reshapeFeat2 = np.reshape(reshapeFeat,(1,-1))
        
        counter += 1
        stdout.write("\r%d" % counter)
        stdout.flush()
        result[i,:] = reshapeFeat2
        
    print result.shape
    
    resultSave = scipy.sparse.csr_matrix(result)
    resultSave32 = resultSave.astype('float32')
    
    if fname == 'train':
        np.savez(dataPath + 'coco_feature.' + fname + experimentPrefix, data=resultSave32.data, indices=resultSave32.indices, indptr=resultSave32.indptr, shape=resultSave.shape)
    else:
        fileName = open(dataPath + 'coco_feature.' + fname + experimentPrefix + '.pkl','wb')
        pickle.dump(resultSave32, fileName ,-1)
        fileName.close()