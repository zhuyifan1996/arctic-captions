import sys
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
from preprocess_util import preprocess_image

# Has to use Absolute Path.
caffe_root        = "/home/gy46/caffe/"
data_path         = "../data/"
text_path         = data_path  + "Flickr8k_text/"
flickr_image_path = data_path  + 'Flicker8k_Dataset/'
annotation_path   = data_path  + 'Flickr8k_text/Flickr8k.lemma.token.txt'
vgg_deploy_path   = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path    = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'
feat_path         = 'feat/flickr8k'

cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)

# Let's make dictionary

"""
annotations['image'] is  <image name>#<image number> where <image number> range from 0 - 5
annotations['caption'] is the caption of each images (five captions per image)
""" 
annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
captions = annotations['caption'].values # all the captions, five captions per image
words = nltk.FreqDist(' '.join(captions).split()).most_common()
wordsDict = {words[i][0]:i+2 for i in range(len(words))}
with open('data/flickr8k/dictionary.pkl', 'wb') as f:
    cPickle.dump(wordsDict, f)

"""
Clean up the data, now that annotations['image'] only contains [<absolute path of the image>]
"""
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

images = pd.Series(annotations['image'].unique())                # list of unique absolute path to image
image_id_dict = pd.Series(np.array(images.index), index=images)  # reverse mapping from idx in [images] to absolute image path in [images]

# (Seems useless) list of ids correspondings to the caption 
caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values

def gen_set_on_index(idx_set):
    """
    Given an index set, return the captions and images set.
    """
    # Select images and captions
    idx_set_ext = [i for idx in idx_set for i in xrange(idx*5, (idx*5)+5)]
    img_set = images[idx_set]
    cap_set = captions[idx_set_ext]

    # Reindex the images
    img_set.index = range(len(idx_set))    
    img_id_dict   = pd.Series(np.array(img_set.index), index=img_set)    # Reverse index

    # list of image ids corresponding to each caption
    cap_img_id = [img_id_dict[img] for img in img_set for i in xrange(5)]
    # create tuples of caption and image id
    caps = zip(cap_set, cap_img_id)
    return caps, img_set

def random_split(tr_size = 6000, te_test=1000): # dev_size = 1000
    """
    Generate random split, take in [tr_size], [te_size], and [dev_size],
    Precondition: [tr_size] + [te_size] < len(images), which is 8000 in this case
    Return :
        (tr_img, tr_caps), (te_img, te_caps), (val_img, val_caps)
    """
    # split up into train, test, and dev
    dev_size = len(images) - tr_size - te_size;
    all_idx = range(len(images))
    np.random.shuffle(all_idx)
    train_idx = all_idx[:(tr_size+1)]
    test_idx = all_idx[(tr_size+1):(tr_size+te_size+1)]
    dev_idx = all_idx[(tr_size+te_size+1):]
    
    return gen_set_on_index(train_idx), gen_set_on_index(test_idx), gen_set_on_index(dev_idx)

def predef_split():
    """
    Load the predefined slipts from Flicker8k Data Text.
    Return:
        (tr_img, tr_caps), (te_img, te_caps), (val_img, val_caps)
    """
    with open(text_path + "Flickr_8k.devImages.txt") as f:
        dev_images = [img.strip() for img in f]

    with open(text_path + "Flickr_8k.testImages.txt") as f:
        test_images = [img.strip() for img in f]
    
    with open(text_path + "Flickr_8k.trainImages.txt") as f:
        train_images = [img.strip() for img in f]

    dev_idx      = [image_id_dict[img] for img in dev_images]
    test_idx     = [image_id_dict[img] for img in test_images]
    train_images = [image_id_dict[img] for img in train_images]
    
    return gen_set_on_index(train_idx), gen_set_on_index(test_idx), gen_set_on_index(dev_idx)

TRAIN_SIZE = 6000
TEST_SIZE  = 1000
# (cap_train, images_train), (cap_test, images_test),(cap_dev, images_dev) = random_split()
(cap_train, images_train), (cap_test, images_test),(cap_dev, images_dev) = predef_split()
preprocess_image(cnn, cap_train, images_train, 'data/flickr8k/flicker_8k_align.train.pkl')
preprocess_image(cnn, cap_test,  images_test,  'data/flickr8k/flicker_8k_align.test.pkl')
preprocess_image(cnn, cap_dev,   images_dev,   'data/flickr8k/flicker_8k_align.dev.pkl')
