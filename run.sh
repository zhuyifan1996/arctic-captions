#! /bin/bash

#generate caps for flickr8k dev/test set
# python generate_caps.py  -d dev model/flickr8k/flickr8k_deterministic_model.npz result2/$(date +%Y-%m-%d:%H:%M:%S)

#python generate_caps.py  -d dev model/flickr8k/flickr8k_deterministic_model.adam64.npz result2/$(date +%Y-%m-%d:%H:%M:%S)
python generate_caps.py  -d dev model/flickr8k/flickr8k_deterministic_model.rmsprop.npz result2/$(date +%Y-%m-%d:%H:%M:%S)

#generate caps for a single, arbitrary image
#python pipeline.py model/flickr8k/flickr8k_deterministic_model.rmsprop.batch256.npz result2/$(date +%Y-%m-%d:%H:%M:%S) ../data/Flicker8k_Dataset/232874193_c691df882d.jpg 

# python pipeline.py model/flickr8k/flickr8k_deterministic_model.npz result2/$(date +%Y-%m-%d:%H:%M:%S) /Users/Grendel/Desktop/ML/textmatters/data/coco/train2014/COCO_train2014_000000411624.jpg 

#python pipeline.py model/flickr8k/flickr8k_deterministic_model.rmsprop.batch256.npz result2/$(date +%Y-%m-%d:%H:%M:%S) /Users/zhuyifan/Documents/2016\ Summer\ Research/textmatters/data/coco/train2014/COCO_train2014_000000411624.jpg 

#train network on flickr8k data
#THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.8 python evaluate_flickr8k.py 

#generate metrics on hypothesis captions
#python metrics.py result2/rmsprop.best.txt data/flickr8k/flicker_8k_refs.dev.txt
