"""
Full pipeline to convert a jpg to a caption.
Steps:
1. Pass the image through a pretrained network, getting a pickle file containing
the image features.
2. Pipe the pickle file through gen_caption.
"""
from cnn_lasagne import CNN_Lasagne
import pandas as pd
import numpy as np
import os
import nltk
import scipy
import json
import cPickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
import pdb

from generate_caps_single import gen_model
from multiprocessing import Process, Queue

from capgen import build_sampler, gen_sample, \
                   load_params, \
                   init_params, \
                   init_tparams, \
                   get_dataset \

# Has to use Absolute Path.
# caffe_root = "/Users/Grendel/caffe/"

annotation_path = '../data/Flickr8k_text/Flickr8k.token.txt'
# vgg_deploy_path = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
# vgg_model_path  = caffe_root + 'models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'
flickr_image_path = '../data/Flicker8k_Dataset'
DATA_PATH = 'data/flickr8k/'

def main(model, saveto, img_path, n_process = 1, pkl_name = None, k=5, sampling = False, normalize=False,zero_pad=False):
    if pkl_name is None:
        pkl_name = model
    with open('%s.pkl'% pkl_name, 'rb') as f:
        options = pkl.load(f)

    cnn = CNN_Lasagne(batch_size=10,
              width=224,
              height=224)

    print "Generating feature"
    feat_list = cnn.get_features(image_list=[img_path],layers='conv5_3', layer_sizes=[512,14,14])
    feat= scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat_list)))

    #try loading the picklified dev features
    # import flickr8k
    # _, (_, feat), _, _ = flickr8k.load_data(load_train=False, load_dev=True, load_test=False)

    #feat = [scipy.sparse.csr_matrix(feat_list[0].flatten())]
    print "Done...\n "

    print "Loading dictionary"
    with open(DATA_PATH+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)
    #invert the dictionary
    print "Dicionary has size {}.".format(len(worddict))
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    print "Done..."

    print 'Generating caption'

    # index -> words
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict[w])
            capsw.append(' '.join(ww))
        return capsw

    def _process_examples(contexts):
        caps = [None] * contexts.shape[0]
        for idx, ctx in enumerate(contexts):
            cc = ctx.todense().reshape([14*14,512])
            if zero_pad:
                cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
                cc0[:-1,:] = cc
            else:
                cc0 = cc
            resp = gen_model(idx, cc0, model, options, k, normalize, word_idict, sampling)
            caps[resp[0]] = resp[1]
            print 'Sample ', (idx+1), '/', contexts.shape[0], ' Done'
            print resp[1]
        return caps

    caps=_seqs2words(_process_examples(feat))

    with open(saveto+'.txt', 'w') as f:
        print "Saving to {}".format(f.name)
        print >>f, '\n'.join(caps)
    print 'Done'

    #ending processes
    # for midx in xrange(n_process):
    #     queue.put(None)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1, help="number of processes to use")
    parser.add_argument('model', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('img_path', type=str)

    args = parser.parse_args()
    main(args.model, args.saveto, args.img_path, n_process=args.p)
