# coding: utf-8

'''
Collect all the region definitions (bounding boxes) that we need
'''


from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip
import json
import xml.etree.ElementTree as ET

# for progress bars
from tqdm import tqdm

from itertools import chain

import datetime

import cPickle as pickle

from glob import glob
import os


import sys
sys.path.append('../Utils')
from utils import icorpus_code, plot_labelled_bb
from utils import get_saiapr_bb, mscoco_image_filename, saiapr_image_filename
from utils import get_imagenet_filename

import logging

# How many of the Berkeley region proposals to use
#   (the first BERKELEYMAX of the 90/10 testset)
BERKELEYMAX = 500

def get_berkeley_edgeboxes(image_id):
    BerkeleyBase = '../Data/Images/SAIAPR/Berkeley_rprops/edgeboxes'
    edgebox_path = BerkeleyBase + '/' + str(image_id) + '.txt'
    return np.loadtxt(edgebox_path)


now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print 'starting to preprocess regions...'
print now



## SAIAPR

#execfile('PreProcDefs/saiapr.py')



## SAIAPR, Berkeley EdgeBox proposals

#execfile('PreProcDefs/saiapr_berkeley.py')


## MSCOCO

#execfile('PreProcDefs/mscoco.py')



## MSCOCO, Google Region Proposals

#execfile('PreProcDefs/mscoco_grps.py')



## ImageNet

execfile('PreProcDefs/imagenet.py')



now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '... and we\'re done!'
print now
