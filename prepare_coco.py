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

# Select training images and captions
import json
with open(annotation_tr,'r') as f:
    stdout.write("Loading data from annotations...\n")
    caps_notes = json.load(f)

    stdout.write("Generating images data...\n")
    images_train     = []
    images_train_idx = []
    for info in caps_notes['images']:
        images_train.append(info['file_name'])
        images_train_idx.append(info['id'])

    stdout.write("Generating captions data...\n")
    cap_train = [(note['caption'], images_train_idx.index(note['image_id'])) for note in caps_notes['annotations']]

    stdout.write("Finished generating data...\n")

# # Reindex the training images
# images_train.index = xrange(TRAIN_SIZE)
# image_id_dict_train = pd.Series(np.array(images_train.index), index=images_train)
# # Create list of image ids corresponding to each caption
# caption_image_id_train = [image_id_dict_train[img] for img in images_train for i in xrange(5)]
# # Create tuples of caption and image id
# cap_train = zip(captions_train, caption_image_id_train)

stdout.write("Start extracting features...\n")
for start, end in zip(range(0, len(images_train)+100, 100), range(100, len(images_train)+100, 100)):
#     image_files = images_train[start:end]
    image_files = images_train[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])
    stdout.write("processing images %d to %d\n" % (start, end))
    stdout.flush()

print "Saving indices information: i -> image_id \n"
with open('../data/coco/coco_align.train.indices.pkl', 'wb') as f:
    cPickle.dump(images_train_idx, f)
    
print "Saving features... \n"
with open('../data/coco/coco_align.train.pkl', 'wb') as f:
    cPickle.dump(cap_train, f,-1)
    cPickle.dump(feat_flatten_list_train, f)




