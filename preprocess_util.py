import numpy as np
import scipy
import cPickle
import pdb

def preprocess_image(cnn, cap_set, images_set, save_to):
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

