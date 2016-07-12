# import caffe
import cv2
import numpy as np
import skimage
import ipdb
import os

DATAPATH = os.getcwd() + os.sep + ".." + os.sep + "data" + os.sep
DATASET = "Flicker8k_Dataset"

def crop_image(x, target_height=227, target_width=227):
    print x
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

deploy     = DATAPATH + 'VGG_ILSVRC_16_layers_deploy.prototxt'
model      = DATAPATH + 'VGG_ILSVRC_16_layers.caffemodel'
mean       = DATAPATH + 'ilsvrc_2012_mean.npy'

# class CNN(object):

#     def __init__(self, deploy=deploy, model=model, mean=mean, batch_size=100, width=227, height=227):

#         self.deploy = deploy
#         self.model = model
#         self.mean = mean

#         self.batch_size = batch_size
#         self.net, self.transformer = self.get_net()
#         self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

#         self.width = width
#         self.height = height

#     def get_net(self):
#         caffe.set_mode_gpu()
#         net = caffe.Net(self.deploy, self.model, caffe.TEST)

#         transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
#         transformer.set_transpose('data', (2,0,1))
#         transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
#         transformer.set_raw_scale('data', 255)
#         transformer.set_channel_swap('data', (2,1,0))

#         return net, transformer

#     def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
#         iter_until = len(image_list) + self.batch_size
#         all_feats = np.zeros([len(image_list)] + layer_sizes)

#         for start, end in zip(range(0, iter_until, self.batch_size), \
#                               range(self.batch_size, iter_until, self.batch_size)):

#             image_batch_file = image_list[start:end]
#             image_batch = np.array(map(lambda x: crop_image(x, target_width=self.width, target_height=self.height), image_batch_file))

#             caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)


#             for idx, in_ in enumerate(image_batch):
#                 caffe_in[idx] = self.transformer.preprocess('data', in_)

#             out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
#             feats = out[layers]

#             all_feats[start:end] = feats

#         return all_feats



import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

from scipy import misc

class CNN_Lasagne(object):
    def __init__(self, deploy=deploy, model=model, mean=mean, batch_size=100, width=227, height=227):
        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.batch_size = batch_size
        self.net, self.out_layer = self.get_net()
        self.load_param(self.out_layer, DATAPATH+"vgg16.pkl")

        self.width = width
        self.height = height


    def get_net(self):
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(
            net['input'], 64, 3, pad=1 )
        net['conv1_2'] = ConvLayer(
            net['conv1_1'], 64, 3, pad=1 )
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(
            net['pool1'], 128, 3, pad=1 )
        net['conv2_2'] = ConvLayer(
            net['conv2_1'], 128, 3, pad=1 )
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(
            net['pool2'], 256, 3, pad=1 )
        net['conv3_2'] = ConvLayer(
            net['conv3_1'], 256, 3, pad=1 )
        net['conv3_3'] = ConvLayer(
            net['conv3_2'], 256, 3, pad=1 )
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(
            net['pool3'], 512, 3, pad=1 )
        net['conv4_2'] = ConvLayer(
            net['conv4_1'], 512, 3, pad=1 )
        net['conv4_3'] = ConvLayer(
            net['conv4_2'], 512, 3, pad=1 )
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(
            net['pool4'], 512, 3, pad=1 )
        net['conv5_2'] = ConvLayer(
            net['conv5_1'], 512, 3, pad=1 )
        net['conv5_3'] = ConvLayer(
            net['conv5_2'], 512, 3, pad=1 )
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(
            net['fc7_dropout'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)
        output_layer = net['prob']

        return net, net['prob'] # the whole net and the output layer

    def load_param(self, output_layer, pickle_file):
        """
        load the model and parameters from specified pickle_file into the lasagne layers,
        with output layer set to [output_layer]
        """
        import pickle

        model = pickle.load(open(pickle_file))
        CLASSES = model['synset words']
        # MEAN_IMAGE = model['mean value']  # right now do not subtract mean image during testing

        lasagne.layers.set_all_param_values(output_layer, model['param values'])

    def prepare_image(self, im):
        import io
        import skimage.transform

        # Resize so smallest dim = 256, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]
        
        rawim = np.copy(im).astype('uint8')
        
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        
        # Convert to BGR
        im = im[::-1, :, :]

        # WARNING: DID NOT subtract mean during testing
        # im = im - MEAN_IMAGE 

        return rawim, floatX(im[np.newaxis])

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        """
        run the pretrained network on the image and extract the output from [layer].
        TODO: currently pipes the image one-by-one
        """
        all_feats = np.zeros([len(image_list)] + layer_sizes)
        for idx, f in enumerate(image_list):
            img = misc.imread(f)
            _, im = self.prepare_image(img)
            all_feats[idx,...] = np.array(lasagne.layers.get_output(self.net[layers], im, deterministic=True).eval())
            # assert feature.shape == (1, 512, 14, 14)
        return all_feats

if __name__=='__main__':
    #testing construction of a CNN
    cnn = CNN_Lasagne(batch_size=10,
              width=224,
              height=224)