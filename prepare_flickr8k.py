import sys
codegit_root = '/home/intuinno/codegit'

sys.path.insert(0, codegit_root)
from cnn_util import CNN
import pandas as pd
import numpy as np
import os
import nltk
import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
import pdb

# Has to use Absolute Path.
caffe_root = "/Users/Grendel/caffe/"
annotation_path = '../data/Flickr8k_text/Flickr8k.token.txt'
vgg_deploy_path = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path  = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'
flickr_image_path = '../data/Flicker8k_Dataset'
feat_path='feat/flickr8k'

def my_tokenizer(s):
    return s.split()

cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=10,
          width=224,
          height=224)

# Let's make dictionary
annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
captions = annotations['caption'].values
words = nltk.FreqDist(' '.join(captions).split()).most_common()
wordsDict = {words[i][0]:i+2 for i in range(len(words))}
with open('data/flickr8k/dictionary.pkl', 'wb') as f:
    cPickle.dump(wordsDict, f)

annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))
images = pd.Series(annotations['image'].unique())
image_id_dict = pd.Series(np.array(images.index), index=images)

caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

# split up into train, test, and dev
TRAIN_SIZE = 6000
TEST_SIZE  = 1000
DEV_SIZE = len(images) - TRAIN_SIZE - TEST_SIZE

print(DEV_SIZE)

# DEV_SIZE = 1000
all_idx = range(len(images))
np.random.shuffle(all_idx)
train_idx = all_idx[0:TRAIN_SIZE]
train_ext_idx = [i for idx in train_idx for i in xrange(idx*5, (idx*5)+5)]
test_idx = all_idx[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
test_ext_idx = [i for idx in test_idx for i in xrange(idx*5, (idx*5)+5)]
dev_idx = all_idx[TRAIN_SIZE+TEST_SIZE:TRAIN_SIZE+TEST_SIZE+DEV_SIZE]
dev_ext_idx = [i for idx in dev_idx for i in xrange(idx*5, (idx*5)+5)]

## TRAINING SET

# Select training images and captions
print(len(images))
print(len(captions))
images_train = images[train_idx]
captions_train = captions[train_ext_idx]

# Reindex the training images
images_train.index = xrange(TRAIN_SIZE)
image_id_dict_train = pd.Series(np.array(images_train.index), index=images_train)
# Create list of image ids corresponding to each caption
caption_image_id_train = [image_id_dict_train[img] for img in images_train for i in xrange(5)]
# Create tuples of caption and image id
cap_train = zip(captions_train, caption_image_id_train)

for start, end in zip(range(0, len(images_train)+100, 100), range(100, len(images_train)+100, 100)):
    image_files = images_train[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d" % (start, end)

with open('data/flickr8k/flicker_8k_align.train.pkl', 'wb') as f:
    cPickle.dump(cap_train, f,-1)
    cPickle.dump(feat_flatten_list_train, f)
    pdb.set_trace()

## TEST SET

# Select test images and captions
images_test = images[test_idx]
captions_test = captions[test_ext_idx]

# Reindex the test images
images_test.index = xrange(TEST_SIZE)
image_id_dict_test = pd.Series(np.array(images_test.index), index=images_test)
# Create list of image ids corresponding to each caption
caption_image_id_test = [image_id_dict_test[img] for img in images_test for i in xrange(5)]
# Create tuples of caption and image id
cap_test = zip(captions_test, caption_image_id_test)

for start, end in zip(range(0, len(images_test)+100, 100), range(100, len(images_test)+100, 100)):
    image_files = images_test[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_test = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_test = scipy.sparse.vstack([feat_flatten_list_test, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d" % (start, end)

with open('data/flickr8k/flicker_8k_align.test.pkl', 'wb') as f:
    cPickle.dump(cap_test, f)
    cPickle.dump(feat_flatten_list_test, f)

## DEV SET

# Select dev images and captions
images_dev = images[dev_idx]
captions_dev = captions[dev_ext_idx]

# Reindex the dev images
images_dev.index = xrange(DEV_SIZE)
image_id_dict_dev = pd.Series(np.array(images_dev.index), index=images_dev)
# Create list of image ids corresponding to each caption
caption_image_id_dev = [image_id_dict_dev[img] for img in images_dev for i in xrange(5)]
# Create tuples of caption and image id
cap_dev = zip(captions_dev, caption_image_id_dev)

for start, end in zip(range(0, len(images_dev)+100, 100), range(100, len(images_dev)+100, 100)):
    image_files = images_dev[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_dev = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_dev = scipy.sparse.vstack([feat_flatten_list_dev, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d" % (start, end)

with open('data/flickr8k/flicker_8k_align.dev.pkl', 'wb') as f:
    cPickle.dump(cap_dev, f)
    cPickle.dump(feat_flatten_list_dev, f)
