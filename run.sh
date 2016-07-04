#! /bin/bash

python generate_caps.py  -d test model/flickr8k/flickr8k_deterministic_model.npz result2/$(date +%Y-%m-%d)
#THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.8 python evaluate_flickr8k.py 
