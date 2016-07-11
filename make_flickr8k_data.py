from cnn_util import *
import pandas as pd
import numpy as np
import os
import scipy
import ipdb
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
import pdb

SMALL_DATA_SET = False
DATAPATH = os.getcwd() + os.sep + ".." + os.sep + "data" + os.sep

annotation_path     = DATAPATH + 'Flickr8k_text' + os.sep + 'Flickr8k.lemma.token.txt'
train_image_list    = DATAPATH + 'Flickr8k_text' + os.sep + 'Flickr_8k.trainImages.txt'
test_image_list     = DATAPATH + 'Flickr8k_text' + os.sep + 'Flickr_8k.testImages.txt'
dev_image_list      = DATAPATH + 'Flickr8k_text' + os.sep + 'Flickr_8k.devImages.txt'

vgg_deploy_path     = DATAPATH + 'VGG_ILSVRC_16_layers_deploy.prototxt'
vgg_model_path      = DATAPATH + 'VGG_ILSVRC_16_layers.caffemodel'
flickr_image_path   = DATAPATH + 'Flicker8k_Dataset'
feat_path           = DATAPATH + 'Flicker8k_features'

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])

annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
all_image = annotations['image'].map(lambda x: x.split('#')[0])
all_image = all_image.unique()
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path, x.split('#')[0]))

captions = annotations['caption'].values

vectorizer = CountVectorizer(lowercase=False).fit(captions)
dictionary = vectorizer.vocabulary_
dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
dictionary = dictionary_series.to_dict()

with open(DATAPATH + 'dictionary.pkl', 'w+') as f:
    cPickle.dump(dictionary, f)

images = pd.Series(annotations['image'].unique())
# ipdb.set_trace()
image_id_dict = pd.Series(np.array(images.index), index=images)

caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

# load train, test and dev
train_images = pd.read_table(train_image_list, sep='\t', header=None, names=['image'])
train_image = train_images['image']
train_idx = train_image.map(lambda x: np.where(all_image==x)[0][0])

# ipdb.set_trace()
test_images = pd.read_table(test_image_list, sep='\t', header=None, names=['image'])
test_image = test_images['image']
test_idx = test_image.map(lambda x: np.where(all_image==x)[0][0])

dev_images = pd.read_table(dev_image_list, sep='\t', header=None, names=['image'])
dev_image = dev_images['image']
dev_idx = dev_image.map(lambda x: np.where(all_image==x)[0][0])

# Only take the first couple images from the train/test/dev sets
if SMALL_DATA_SET:
    train_idx = train_idx[0:(min(len(train_idx), 10))]
    dev_idx   = train_idx[0:(min(len(dev_idx), 10))]
    test_idx  = dev_idx[0:(min(len(test_idx), 10))]

cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=2,
          width=224,
          height=224)

#training set
images_train = images[train_idx]
image_id_dict_train = image_id_dict[train_idx]
caption_image_id_train = caption_image_id[train_idx]
captions_train = captions[train_idx]
cap_train = zip(captions_train, caption_image_id_train)

for start, end in zip(range(0, len(images_train)+1, 1), range(1, len(images_train)+1, 1)):
    image_files = images_train[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d " % (start, end)


with open(DATAPATH + 'flicker_8k_align.train.pkl', 'w+') as f:
    cPickle.dump(cap_train, f)
    cPickle.dump(feat_flatten_list_train, f)

# test set
images_test = images[test_idx]
image_id_dict_test = image_id_dict[test_idx]
caption_image_id_test = caption_image_id[test_idx]
captions_test = captions[test_idx]
cap_test = zip(captions_test, caption_image_id_test)

for start, end in zip(range(0, len(images_test)+1, 1), range(1, len(images_test)+1, 1)):
    image_files = images_test[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_test = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_test = scipy.sparse.vstack([feat_flatten_list_test, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d " % (start, end)

with open(DATAPATH + 'flicker_8k_align.test.pkl', 'w+') as f:
    cPickle.dump(cap_test, f)
    cPickle.dump(feat_flatten_list_test, f)

# dev set
images_dev = images[dev_idx]
image_id_dict_dev = image_id_dict[dev_idx]
caption_image_id_dev = caption_image_id[dev_idx]
captions_dev = captions[dev_idx]
cap_dev = zip(captions_dev, caption_image_id_dev)

for start, end in zip(range(0, len(images_dev)+1, 1), range(1, len(images_dev)+1,  1)):
    image_files = images_dev[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_dev = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_dev = scipy.sparse.vstack([feat_flatten_list_dev, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d " % (start, end)

with open(DATAPATH + 'flicker_8k_align.dev.pkl', 'wb') as f:
    cPickle.dump(cap_dev, f)
    cPickle.dump(feat_flatten_list_dev, f)
