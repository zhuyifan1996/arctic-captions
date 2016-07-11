import pdb
import sys
import scipy
import  cPickle as pickle
import numpy as np
import sys
import json
import caffe
from preprocess_util import preprocess_image
import pandas as pd
import nltk

# Setup
CWD = os.getcwd() + "/"
originalImagesPath     = CWD + '../data/coco/originalImages'
preprocessedImagesPath = CWD + '../data/coco/processedImages/'

caffe_root = '/home/gy46/caffe/'

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

# Make the dictionary form the captions of the training data
with open(annotation_tr,'r') as f:
    stdout.write("Loading data from annotations...\n")
    caps_notes = json.load(f)

    # Make a reverse dictionary from image_id to captions

"""
Generate Dictionary
"""
corpus3 = ' '.join([note['caption'] for note in caps_notes['annotations']])
words = nltk.FreqDist(corpus3.split()).most_common()
wordsDict = {words[i][0]:i+2 for i in range(len(words))}
with open(dataPath + 'dictionary.pkl', 'wb') as f:
    pickle.dump(wordsDict, f)

"""
Extract Features
"""
cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)
files = [ 'val','test','train']
data  = {
    'val'   : {'image' : [], 'captions' : [] },
    'test'  : {'image' : [], 'captions' : [] },
    'train' : {'image' : [], 'captions' : [] }
}

for fname in files:
    # Make the dictionary form the captions of the training data
    with open(annotation_tr,'r') as f:
        caps_notes = json.load(f)
        
        img_dict  = { img['id']:img['file_name'] for img in caps_notes['images'] }
        images    = img_dict.values()
        img_idx   = img_dict.keys()

        def dput(d, img_id, cap):
            if not img_id in d:
               d[img_id] = []
            d[img_id].append(cap)
        cap_dict  = {}
        for note in caps_notes['annotations']:
            dput(cap_dict, note['image_id'], note['caption'])
        
        captions = []
        for img_id, ccp_lst in cap_dict:
            captions += [ (cap, img_idx.index(img_id)) for cap in ccp_lst ] 

    preprocess_image(cnn, captions, images, 'data/coco/coco.'+fname+'.pkl')
