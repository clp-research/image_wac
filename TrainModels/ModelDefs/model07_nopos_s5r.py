# coding: utf-8

'''
Model 07

SAIAPR, 90/10, no rel, min 40, logregL1, 5, random
... but only using *non*positional features

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
    'notes': 'no positional features'
}

basename = 'model07_nopos_s5r'


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


# only leaving the non-positional features
X = X[:,:-7]



### VI. And... train away!

clsf = train_model(srefdf_tr, X, wordlist,
                   (linear_model.LogisticRegression, {'penalty':'l1'}),
                   nneg=model['nneg'], nsrc=model['nsrc'])

with gzip.open('../TrainedModels/' + basename + '.pklz', 'w') as f:
    pickle.dump(clsf, f)

with open('../TrainedModels/' + basename + '.json', 'w') as f:
    json.dump(model, f)

print strftime("%Y-%m-%d %H:%M:%S")
