# coding: utf-8

'''
Compute the feature representations for the image regions

NEVER ever import from here! This is a script that is meant to be run!
When imported, it will run, and potentially overwrite stuff!
(Yes, I know, there should be a main function...)
'''

from __future__ import division

#import json
#import os

import numpy as np
#import pandas as pd

from tqdm import tqdm

from time import strftime

import cPickle as pickle
import gzip

#import re
#import scipy.io
#from itertools import chain

from PIL import Image as PImage

import matplotlib.pyplot as plt
#import matplotlib

from glob import glob



from sklearn_theano.feature_extraction import GoogLeNetTransformer
from sklearn_theano.feature_extraction.caffe.googlenet_layer_names import get_googlenet_layer_names



import sys
sys.path.append('../Utils')
from utils import code_icorpus
from utils import get_thumbnail
from utils import join_imagenet_id


BASETEMPLATE_TMP = 'ExtrFeatsOut/Temp/%s_%03d.pklz'
BASETEMPLATE_TMP_GLOB = 'ExtrFeatsOut/Temp/%s_*.pklz' # oh dear
BASETEMPLATE_FIN = 'ExtrFeatsOut/%s.npz'


# how many images are loaded up into on X to pass to extractor
#  (which then has its own batchsize, for the conv net)
img_batch_size = 6000


# Structure:
# - load up required DFs
# - cycle through corpus (DF), creating batch files
# - assemble joint file
# - profit






def compute_posfeats(img, bb):
    ih, iw, _ = img.shape
    x,y,w,h = bb
    # x1, relative
    x1r = x / iw
    # y1, relative
    y1r = y / ih
    # x2, relative
    x2r = (x+w) / iw
    # y2, relative
    y2r = (y+h) / ih
    # area
    area = (w*h) / (iw*ih)
    # ratio image sides (= orientation)
    ratio = iw / ih
    # distance from center (normalised)
    cx = iw / 2
    cy = ih / 2
    bcx = x + w / 2
    bcy = y + h / 2
    distance = np.sqrt((bcx-cx)**2 + (bcy-cy)**2) / np.sqrt(cx**2+cy**2)
    # done!
    return np.array([x1r,y1r,x2r,y2r,area,ratio,distance])


def compute_feats(bbdf):
    X_pos = []
    X_i = []
    ids = []
    file_counter = 1
    prev_iid, prev_img = (None, None)
    # FIXME, for debugging only! Reduced size or starting with offset
    # bbdf = bbdf[28524:]  # bbdf[54000:]
    for n, row in tqdm(bbdf.iterrows(), total=len(bbdf)):
        this_icorpus = row['i_corpus']
        this_image_id = row['image_id']
        this_region_id = row['region_id']
        this_bb = row['bb']
        # 2016-04-08: as note for future: When extracting
        #  feats for imagenet regions, must
        #  - create combined filename out of image_id and region_id
        #  - neutralise positional features, by setting bb given
        #    to it to 0,0,w,h. So that all ImageNet regions
        #    end up with same positions.
        if code_icorpus[this_icorpus] == 'image_net':
            this_image_id_mod = join_imagenet_id(this_image_id,
                                                 this_region_id)
            this_bb_mod = [0,0,this_bb[2],this_bb[3]]
        else:
            this_image_id_mod = this_image_id
            this_bb_mod = this_bb

        if np.min(this_bb_mod[2:]) <= 0:
            print 'skipping over this image (%s,%d). 0 bb! %s' % \
                (code_icorpus[this_icorpus], this_image_id, str(this_bb_mod))
            continue

        (prev_iid, prev_img), img_resized = \
                    get_thumbnail((prev_iid, prev_img), 
                      this_icorpus, this_image_id_mod, this_bb)


        if len(prev_img.shape) != 3 or \
             (len(prev_img.shape) == 3 and prev_img.shape[2] != 3):
            print 'skipping over this image (%s,%d). b/w?' % \
                (code_icorpus[this_icorpus], this_image_id)
            continue
        # If we continue below this line, getting region worked
        X_i.append(img_resized)
        this_pos_feats = compute_posfeats(prev_img, this_bb_mod)
        X_pos.append(this_pos_feats)
        ids.append(np.array([this_icorpus, this_image_id, this_region_id]))

        # is it time to do the actual extraction on this batch
        #  and write out to disk?
        if (n+1) % img_batch_size == 0:
            filename = BASETEMPLATE_TMP % (code_icorpus[this_icorpus], 
                                           file_counter)
            print strftime("%Y-%m-%d %H:%M:%S")
            print "new batch!", n, file_counter, filename
            
            try:
                X = gltr.transform(X_i)
            except ValueError as e:
                print 'Exception! But why? Skipping this whole batch..'
                X_i = []
                ids = []
                X_pos = []
                continue
                #raise e

            X_ids = np.array(ids)
            X_pos = np.array(X_pos)
            print X_ids.shape, X.shape, X_pos.shape
            X_f = np.hstack([X_ids,
                             X, 
                             X_pos])
            with gzip.open(filename, 'w') as f:
               pickle.dump(X_f, f)
            print X_f.shape

            ids = []
            X_pos = []
            X_i = []
            file_counter += 1
    # and back to the for loop

    # we're done, so what we have needs to be processed in any case
    filename = BASETEMPLATE_TMP % (code_icorpus[this_icorpus], 
                                   file_counter)
    print strftime("%Y-%m-%d %H:%M:%S")
    print "final batch!", n, file_counter, filename

    X = gltr.transform(X_i)

    X_ids = np.array(ids)
    X_pos = np.array(X_pos)
    X_f = np.hstack([X_ids,
                     X, 
                     X_pos])
    with gzip.open(filename, 'w') as f:
       pickle.dump(X_f, f)
    print X_f.shape

def concatenate_feat_batches(bbdf):
    this_icorpus_name = code_icorpus[bbdf.ix[0,0]]
    outpath = BASETEMPLATE_FIN % (this_icorpus_name)
    parts = sorted(glob(BASETEMPLATE_TMP_GLOB % (this_icorpus_name)))
    Xs = []
    for part in parts:
        with gzip.open(part, 'r') as f:
            Xs.append(pickle.load(f))
    X = np.concatenate(Xs)
    print '....'
    print outpath
    print 'concatenated shape:'
    print X.shape
    np.savez_compressed(outpath, X)



### ok, run it!

if __name__ == '__main__':
    gltr = GoogLeNetTransformer(force_reshape=True,
                                batch_size=10,
                                output_layers=(get_googlenet_layer_names()[-4],))

    # print '-' * 40
    # print strftime("%Y-%m-%d %H:%M:%S")
    # print "SAIAPR"
    # with gzip.open('../Preproc/PreProcOut/saiapr_bbdf.pklz', 'r') as f:
    #     saia_bbdf = pickle.load(f)
    # compute_feats(saia_bbdf)
    # concatenate_feat_batches(saia_bbdf)


    # print '-' * 40
    # print strftime("%Y-%m-%d %H:%M:%S")
    # print "SAIAPR BERK"
    # with gzip.open('../Preproc/PreProcOut/berkeley_bbdf.pklz', 'r') as f:
    #     saia_berk_bbdf = pickle.load(f)
    # compute_feats(saia_berk_bbdf)
    # concatenate_feat_batches(saia_berk_bbdf)


    # print '-' * 40
    # print strftime("%Y-%m-%d %H:%M:%S")
    # print "MSCOCO"
    # with gzip.open('../Preproc/PreProcOut/mscoco_bbdf.pklz', 'r') as f:
    #     coco_bbdf = pickle.load(f)
    # compute_feats(coco_bbdf)
    # concatenate_feat_batches(coco_bbdf)

    # print '-' * 40
    # print strftime("%Y-%m-%d %H:%M:%S")
    # print "MSCOCO RPROPS"
    # with gzip.open('../Preproc/PreProcOut/cocogrprops_bbdf.pklz', 'r') as f:
    #     grprop_bbdf = pickle.load(f)
    # compute_feats(grprop_bbdf)
    # concatenate_feat_batches(grprop_bbdf)


    print '-' * 40
    print strftime("%Y-%m-%d %H:%M:%S")
    print "ImageNet"
    with gzip.open('../Preproc/PreProcOut/imagenet_bbdf.pklz', 'r') as f:
        in_bbdf = pickle.load(f)
    compute_feats(in_bbdf)
    concatenate_feat_batches(in_bbdf)





    print '-' * 40
    print 'And we\'re done!'
    print strftime("%Y-%m-%d %H:%M:%S")


