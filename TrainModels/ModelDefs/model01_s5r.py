# coding: utf-8

'''
Experiment 1 / Model 1

A model is defined by a number of (hyper)parameters:
- what data is used (which corpus, which split),
- how it is preprocessed (if at all)
- how the training data is assembled (especially the negative examples)
- and which learner is used (including its parameters)

Since it's actually easier to maintain, and almost as easy to write and read,
the model configuration is done here in actual python code rather than via
command line switches or config files.

SAIAPR, 90/10, no rel, min 40, logregL1, 5, random
'''

from __future__ import division
import numpy as np
import gzip
import cPickle as pickle
import json
from time import strftime

from sklearn import linear_model

import sys
sys.path.append('../../Utils')
sys.path.append('..')
from utils import filter_by_filelist

from train_model import train_model
from train_model import filter_relational_expr, wordlist_by_criterion


####### The training pipeline ########

print strftime("%Y-%m-%d %H:%M:%S")


## some parameters are fixed / documented here
##  processing steps still need to be triggered explicitly

model = {
    'rcorp': 'referit',
    'rel':   'excl',
    'wrdl':  'min',
    'wprm':  40,
    'clsf':  'logreg-l1',
    'nneg':  5,
    'nsrc':  'random',
    'notes': ''
}

basename = 'model1_s5r'


### I. Get the refexp data (the refdf)

with gzip.open('../../Preproc/PreProcOut/saiapr_refdf.pklz', 'r') as f:
    srefdf = pickle.load(f)

### II. Set the split to use

with open('../../Preproc/PreProcOut/saiapr_90-10_splits.json', 'r') as f:
    ssplit90 = json.load(f)

srefdf_t = filter_by_filelist(srefdf, ssplit90['train'])

### III. Preprocess: remove relational expressions from training

srefdf_tr = filter_relational_expr(srefdf_t)

### IV. Set list of words to train

wordlist = wordlist_by_criterion(srefdf_tr, model['wrdl'], model['wprm'])

### V. Get the region features

X = np.load('../../ExtractFeats/ExtrFeatsOut/saiapr.npz')
X = X['arr_0']

### VI. And... train away!

clsf = train_model(srefdf_tr, X, wordlist,
                   (linear_model.LogisticRegression, {'penalty':'l1'}),
                   nneg=model['nneg'], nsrc=model['nsrc'])

with gzip.open('../TrainedModels/' + basename + '.pklz', 'w') as f:
    pickle.dump(clsf, f)

with open('../TrainedModels/' + basename + '.json', 'w') as f:
    json.dump(model, f)

print strftime("%Y-%m-%d %H:%M:%S")
