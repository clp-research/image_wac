# coding: utf-8

'''
Run the models on the data sets.
'''


# TODO: Need to write out which region was selected, so that I can do
#  error analysis later!

from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json

import scipy.stats

from time import strftime

import sys
sys.path.append('../TrainModels')
sys.path.append('../Utils')
from utils import filter_by_filelist
from train_model import STOPWORDS, is_relational, create_word2den, make_train
from apply_model import *

from augment_model import compute_confidences

import os



outfilename = 'EvalOut/results-cutoff.pklz'

if os.path.isfile(outfilename):
    print 'Outfile (%s) exists. Better check before I overwrite anything!' % (outfilename)
    exit()



results = []


print strftime("%Y-%m-%d %H:%M:%S")
print 'Loading up data. This may take some time.'

### Load up basic common data

## Corpora, Features

X = np.load('/Volumes/USB_128GB/Models/2016-v3-image-wac/saiapr.npz')
X = X['arr_0']

Xc = np.load('/Volumes/USB_128GB/Models/2016-v3-image-wac/mscoco.npz')
Xc = Xc['arr_0']

Xb = np.load('/Volumes/USB_128GB/Models/2016-v3-image-wac/saiapr_berkeley.npz')
Xb = Xb['arr_0']

Xg = np.load('/Volumes/USB_128GB/Models/2016-v3-image-wac/mscoco_grprops.npz')
Xg = Xg['arr_0']


## Corpora, Refexps (refdfs)

with gzip.open('../Preproc/PreProcOut/saiapr_refdf.pklz', 'r') as f:
    srefdf = pickle.load(f)

with gzip.open('../Preproc/PreProcOut/refcoco_refdf.pklz', 'r') as f:
    rrefdf = pickle.load(f)

with gzip.open('../Preproc/PreProcOut/grex_refdf.pklz', 'r') as f:
    grefdf = pickle.load(f)



## Splits

with open('../Preproc/PreProcOut/saiapr_90-10_splits.json', 'r') as f:
    ssplit90 = json.load(f)

with open('../Preproc/PreProcOut/refcoco_splits.json', 'r') as f:
    rcocosplits = json.load(f)

# rcocosplits.keys(), map(len, rcocosplits.values())
# ([u'testA', u'train', u'val', u'testB'], [750, 16994, 1500, 750])



## Bounding box definitions

with gzip.open('../Preproc/PreProcOut/berkeley_bbdf.pklz', 'r') as f:
    b_bbdf = pickle.load(f)
with gzip.open('../Preproc/PreProcOut/saiapr_bbdf.pklz', 'r') as f:
    s_bbdf = pickle.load(f)
with gzip.open('../Preproc/PreProcOut/cocogrprops_bbdf.pklz', 'r') as f:
    g_bbdf = pickle.load(f)
with gzip.open('../Preproc/PreProcOut/mscoco_bbdf.pklz', 'r') as f:
    c_bbdf = pickle.load(f)



print strftime("%Y-%m-%d %H:%M:%S")
print 'Off we go.'




## Run the evaluations:

# This has full access to all global variables.
#  Most importantly, all the code there will add to results.

# execfile('EvalDefs/01.py')

# execfile('EvalDefs/02.py')

# execfile('EvalDefs/03.py')

# execfile('EvalDefs/04.py')

# execfile('EvalDefs/05.py')

# execfile('EvalDefs/10.py')

# execfile('EvalDefs/11.py')

# execfile('EvalDefs/max_area_baselines.py')

# execfile('EvalDefs/ablations.py')

# execfile('EvalDefs/rprops.py')

# execfile('EvalDefs/rprops2.py')

# execfile('EvalDefs/top20.py')

execfile('EvalDefs/01_cutoff.py')

## Write everything to a file, go home.

with gzip.open(outfilename, 'w') as f:
    pickle.dump(results, f)
