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

# Has to use Absolute Path.
caffe_root        = "/home/gy46/caffe/"
annotation_path   = '../data/Flickr8k_text/Flickr8k.token.txt'
vgg_deploy_path   = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path    = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'
flickr_image_path = '../data/Flicker8k_Dataset'
feat_path         = 'feat/flickr8k'

def my_tokenizer(s):
    return s.split()

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

# Seem useless, from 0-5 per image
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])

"""
Clean up the data, now that annotations['image'] only contains [<absolute path of the image>]
"""
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

images = pd.Series(annotations['image'].unique())                # list of unique absolute path to image
image_id_dict = pd.Series(np.array(images.index), index=images)  # reverse mapping from idx in [images] to absolute image path in [images]

# list of ids correspondings to the caption 
caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
# cap = zip(captions, caption_image_id)

def random_split(tr_size = 6000, te_test=1000):
    """
    Generate random split, take in [tr_size], [te_size], and [dev_size],
    Precondition: [tr_size] + [te_size] < len(images), which is 8000 in this case
    Return :
        (tr_img, tr_caps), (te_img, te_caps), (val_img, val_caps)
    """
	# split up into train, test, and dev
	dev_size = len(images) - tr_size - te_size;
	print(DEV_SIZE)

	# DEV_SIZE = 1000
	all_idx = range(len(images))
	np.random.shuffle(all_idx)
	train_idx = all_idx[:(tr_size+1)]
	test_idx = all_idx[(tr_size+1):(tr_size+te_size+1)]
	dev_idx = all_idx[(tr_size+te_size+1):]

	def gen_set_on_index(idx_set):
		# Select images and captions
		idx_set_ext = [i for idx in idx_set for i in xrange(idx*5, (idx*5)+5]
		img_set = images[idx_set]
		cap_set = captions[idx_set_ext]

		# Reindex the images
		img_set.index = range(len(idx_set))	
		img_id_dict   = pd.Series(np.array(img_set.index), index=img_set)	# Reverse index

		# list of image ids corresponding to each caption
		cap_img_id = [img_id_dict[img] for img in img_set for i in xrange(5)]
		# create tuples of caption and image id
		caps = zip(cap_set, cap_img_id)
		return caps, img_set
	
	return gen_set_on_index(train_idx), gen_set_on_index(test_idx), gen_set_on_index(dev_idx)

def predef_split():
	"""
	Load the predefined slipts from Flicker8k Data Text.
	Return:
        (tr_img, tr_caps), (te_img, te_caps), (val_img, val_caps)
	"""
	

def preprocess_image(cap_set, images_set, save_to):
	for start, end in zip(range(0, len(images_set)+100, 100), range(100, len(images_set)+100, 100)):
		image_files = images_set[start:end]
		feat = cnn.get_features(image_list=image_files, layers='conv5_4', layer_sizes=[512,14,14])
		if start == 0:
			feat_flatten_list = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
		else:
			feat_flatten_list = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

		print "processing images %d to %d" % (start, end)

	with open(save_to, 'wb') as f:
		cPickle.dump(cap_set, f,-1)
		cPickle.dump(feat_flatten_list, f)
		pdb.set_trace()

TRAIN_SIZE = 6000
TEST_SIZE  = 1000
(cap_train, images_train), (cap_test, images_test),(cap_dev, images_dev) = random_split()
preprocess_image(cap_train, images_train, 'data/flickr8k/flicker_8k_align.train.pkl')
preprocess_image(cap_test,  images_test,  'data/flickr8k/flicker_8k_align.test.pkl')
preprocess_image(cap_dev,   images_dev,   'data/flickr8k/flicker_8k_align.dev.pkl')
