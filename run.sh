#! /bin/bash
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.8 python evaluate_flickr8k.py 
