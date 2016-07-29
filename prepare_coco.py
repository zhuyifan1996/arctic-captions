from cnn_util import CNN
import pdb
import os
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

caffe_root = '/home/yz542/caffe/'

vgg_deploy_path = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path  = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'

dataPath        = CWD + '../data/coco/'
annotation_path = dataPath + 'annotations/'

image_paths     = {
    'train'     : dataPath + 'train2014/',
    'test'      : dataPath + 'test2014/',
    'val'       : dataPath + 'val2014/'
}

# Make the dictionary form the captions of the training data
with open(os.path.join(annotation_path, 'captions_train2014.json'),'r') as f:
    print "Loading data from annotations...\n"
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
          batch_size=32,
          width=224,
          height=224)
#files = [ 'val','test','train']
files = [ 'val','train']
data  = {
    'val'   : {'image' : [], 'captions' : [] },
    'test'  : {'image' : [], 'captions' : [] },
    'train' : {'image' : [], 'captions' : [] }
}

for fname in files:
    # Make the dictionary form the captions of the training data
    with open(os.path.join(annotation_path, 'captions_'+fname+'2014.json'),'r') as f:
        print "Loading caption information..."
        caps_notes = json.load(f)
        print "Done."

        print "Making image dictionaries..."
        img_dict  = { img['id']:img['file_name'] for img in caps_notes['images'] }
        images    = img_dict.values()
        img_idx   = img_dict.keys()
        print "Done."

        print "Making caption dictionaries..."
        cap_dict={}
        for note in caps_notes['annotations']:
            img_id = note['image_id']
            if not img_id in cap_dict:
               cap_dict[img_id] = []
            cap_dict[img_id].append(note['caption'])
        print "Done."

        print "Making caption batch..." 
        captions = []
        for img_id, ccp_lst in cap_dict.iteritems():
            captions += [ (cap, img_idx.index(img_id)) for cap in ccp_lst ] 
        print "Done."

        print "Making images batches..."
        img_abs_paths = [ os.path.join(image_paths[fname], str(img)) for img in images]
        print 'Done.' 

    print "Start preprocessing images..."
    preprocess_image(cnn, captions, img_abs_paths, 'data/coco/coco.'+fname+'.pkl')
    print "Done."
