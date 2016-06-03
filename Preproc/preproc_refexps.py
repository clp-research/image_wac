# coding: utf-8
from __future__ import division

import json
import numpy as np
import pandas as pd
import cPickle as pickle
import gzip
import re
import datetime

import nltk
from nltk.tag.perceptron import PerceptronTagger

import sys
sys.path.append('../Utils')

from utils import icorpus_code


def preproc(utterance):
    utterance = re.sub('[\.\,\?;]+', '', utterance)
    return utterance.lower()
preproc_vec = np.vectorize(preproc)

tagger = PerceptronTagger()
def postag(refexp):
    return nltk.tag._pos_tag(nltk.word_tokenize(refexp), None, tagger)


now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print 'starting to preprocess...'
print now

## ReferIt SAIAPR
referitpath = '../Data/RefExps/SAIAPR/ReferIt/RealGames.txt'

refdf = pd.read_csv(referitpath, sep='~', names=['ID', 'refexp', 'regionA', 'regionB'])
refdf['file'] = refdf['ID'].apply(lambda x: int(x.split('.')[0].split('_')[0]))
refdf['region'] = refdf['ID'].apply(lambda x: int(x.split('.')[0].split('_')[1]))
refdf['refexp'] = preproc_vec(refdf['refexp'])

refdf['i_corpus'] = icorpus_code['saiapr']
refdf['r_corpus'] = 'referit'
refdf['image_id'] = refdf['file']
refdf['region_id'] = refdf['region']
refdf['rex_id'] = refdf.index.tolist()

refdf['tagged'] = refdf['refexp'].apply(postag)

refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus', 
               'rex_id', 'refexp', 'tagged']]


# load and write out the splits on SAIAPR as used by Berkeley group (50/50)
b_splits_train_p = '../Data/Images/SAIAPR/Berkeley_rprops/referit_trainval_imlist.txt'
b_splits_test_p = '../Data/Images/SAIAPR/Berkeley_rprops/referit_test_imlist.txt'

saiapr_train_files = np.loadtxt(b_splits_train_p, dtype=int)
saiapr_test_files = np.loadtxt(b_splits_test_p, dtype=int)

saiapr_berkeley_splits = {
    'test': list(saiapr_test_files),
    'train': list(saiapr_train_files)
}


with open('PreProcOut/saiapr_berkeley_10-10_splits.json', 'w') as f:
    json.dump(saiapr_berkeley_splits, f)


# create a 90/10 split as well, to have more training data
saiapr_train_90 = list(saiapr_train_files) + list(saiapr_test_files)[:8000]
saiapr_test_90 = list(saiapr_test_files)[8000:]
saiapr_90_10_splits = {
    'test': saiapr_test_90,
    'train': saiapr_train_90
}
with open('PreProcOut/saiapr_90-10_splits.json', 'w') as f:
    json.dump(saiapr_90_10_splits, f)





## ReferIt COCO
refcoco_path = '../Data/RefExps/MSCOCO/ReferIt_COCO/cleaned(licheng).p'
with open(refcoco_path, 'r') as f:
    refcoco = pickle.load(f)

refcocodf = pd.DataFrame(refcoco)

refdf_list = []
for n, this_row in refcocodf.iterrows():
    this_file = this_row['image_id']
    this_region = this_row['ann_id']
    for this_sentence, this_rexid in zip(this_row['sentences'],
                                         this_row['sent_ids']):
        this_sentence_sent = this_sentence['sent']
        refdf_list.append((this_file, this_region,
                           this_sentence_sent, this_rexid))
        
refcocodf_tmp = pd.DataFrame(refdf_list, 
                             columns='image_id region_id refexp rex_id'.split())

refcocodf_tmp['i_corpus'] = icorpus_code['mscoco']
refcocodf_tmp['r_corpus'] = 'refcoco'

refcocodf_tmp['tagged'] = refcocodf_tmp['refexp'].apply(postag)
refcoco_fin = refcocodf_tmp[['i_corpus', 'image_id', 'region_id', 
                             'r_corpus', 'rex_id', 'refexp', 'tagged']]


# write out the suggested splits from ReferIt team
#  here we have more than just train and val
refcoco_splits = {}
for part in refcocodf['split'].value_counts().index:
    this_filelist = list(set(refcocodf[refcocodf['split'] == part]['image_id'].tolist()))
    # print len(this_filelist)
    refcoco_splits[part] = this_filelist

with open('PreProcOut/refcoco_splits.json', 'w') as f:
    json.dump(refcoco_splits, f)



## Google_Refexp COCO
gjson_p = '../Data/RefExps/MSCOCO/google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
with open(gjson_p, 'r') as f:
    gexp = json.load(f)
gexan = pd.DataFrame(gexp['annotations']).T
gexrex = pd.DataFrame(gexp['refexps']).T

gjson_p = '../Data/RefExps/MSCOCO/google_refexp_dataset_release/google_refexp_val_201511_coco_aligned.json'
with open(gjson_p, 'r') as f:
    gexpv = json.load(f)
gexanv = pd.DataFrame(gexpv['annotations']).T
gexrexv = pd.DataFrame(gexpv['refexps']).T

gexanfull = pd.concat([gexan, gexanv])
gexrexfull = pd.concat([gexrex, gexrexv])


outrows = []
for n, row in gexanfull.iterrows():
    this_image_id = row['image_id']
    this_anno_id = row['annotation_id']
    this_refexp_ids = row['refexp_ids']
    for this_refexp_id in this_refexp_ids:
        this_refexp = gexrexfull[gexrexfull['refexp_id']                                 == this_refexp_id]['raw'][0]
        this_refexp = re.sub('[\.\,\?;]+', '', this_refexp).lower()
        this_refexp = this_refexp.encode('UTF-8')
        outrows.append((this_image_id, this_anno_id,
                        this_refexp, this_refexp_id))
        
gexdf = pd.DataFrame(outrows,
                     columns='image_id region_id refexp rex_id'.split())
gexdf['i_corpus'] = icorpus_code['mscoco']
gexdf['r_corpus'] = 'grex'

gexdf['tagged'] = gexdf['refexp'].apply(postag)

gexdf = gexdf[['i_corpus', 'image_id', 'region_id',
               'r_corpus', 'rex_id', 'refexp', 'tagged']]



# write out the splits as suggested by Google team
#   NB: The splits here contain *refexp_ids*, not image_ids!
gexsplits = {
    'train': gexrex['refexp_id'].tolist(),
    'val': gexrexv['refexp_id'].tolist()
    }

with open('PreProcOut/google_refexp_rexsplits.json', 'w') as f:
    json.dump(gexsplits, f)



### Done!

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print 'done with the assembling...'
print now

print 'example rows:'
print gexdf.head(1)
print refdf.head(1)
print refcoco_fin.head(1)

print 'refdf: %d\nrefcoco: %d\ngex: %d' % (len(refdf), 
                                           len(refcoco_fin), len(gexdf))




## Well ok, we should probably write out to disk as well:

with gzip.open('PreProcOut/saiapr_refdf.pklz', 'w') as f:
    pickle.dump(refdf, f)
    
with gzip.open('PreProcOut/refcoco_refdf.pklz', 'w') as f:
    pickle.dump(refcoco_fin, f)
    
with gzip.open('PreProcOut/grex_refdf.pklz', 'w') as f:
    pickle.dump(gexdf, f)


now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '.. and done!'
print now


