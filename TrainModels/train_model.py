# coding: utf-8

from __future__ import division
import numpy as np

from collections import Counter

RELWORDS = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'front of',
            'right of',
            'left of',
            'ontop of',
            'next to',
            'middle of']

STOPWORDS = ['the', 'a', 'an']


def is_relational(expr):
    for rel in RELWORDS:
        if rel in expr:
            return True
    return False

def filter_relational_expr(refdf):
    '''View on given refdf with only non-relation refexps.'''
    return refdf[~(refdf['refexp'].apply(is_relational))]


def wordlist_min_freq(refdf, minfreq, stopped=True):
    '''Wordlist out of refdf; minimum frequency criterion.'''
    rexc = Counter(' '.join(refdf['refexp'].tolist()).split())
    list_ = [w for w,c in rexc.items() if c >= minfreq]
    if stopped:
        list_ = [w for w in list_ if w not in STOPWORDS]
    return list_

def wordlist_n_top(refdf, ntop, stopped=True):
    '''Wordlist out of refdf; n-most frequent criterion.'''
    rexc = Counter(' '.join(refdf['refexp'].tolist()).split())
    if stopped:
        list_ = rexc.most_common(ntop + len(STOPWORDS))
        list_ = [w for w,_ in list_ if w not in STOPWORDS][:ntop]
    else:
        list_ = [w for w,_ in rexc.most_common(ntop)]
    return list_

def wordlist_by_criterion(refdf, criterion, parameter, stopped=True):
    if criterion == 'min':
        return wordlist_min_freq(refdf, parameter, stopped=stopped)
    if criterion == 'ntop':
        return wordlist_n_top(refdf, parameter, stopped=stopped)


def create_word2den(refdf, wordlist, tagged=False):
    '''Given refdf and wordlist, returns dict of occurences (id triples) 
    of words.'''
    word2den = {}
    for _, row in refdf.iterrows():
        exprlist = row['refexp'].split()
        if tagged:
            exprlist = row['tagged']
        for word in exprlist:
            if word in wordlist:
                word_den_list = []
                word_den_list = word2den.get(word, [])
                word_den_list.append((row['i_corpus'],
                                      row['image_id'],
                                      row['region_id']))
                word2den[word] = word_den_list
    return word2den

def make_train(X, wrd2dn, word, nneg, nsrc):
    '''Construct training feature set for word.'''
    X_train = []
    y_train = []
    for _, this_image_id, this_region_id in wrd2dn[word]:
        # pos example
        pos_feats = X[np.logical_and(X[:, 1] == this_image_id,
                                     X[:, 2] == this_region_id)][:, 3:]
        if len(pos_feats) != 1:
            print 'more than one feature vec for this region (%d, %d)!' \
                % (this_image_id, this_region_id)
            print 'skipping all of it'
            #print 'taking first of it!'
            #pos_feats = pos_feats[0]
            continue

        X_train.append(pos_feats)
        y_train.append(True)
        # neg examples

        # Take negative examples from same image:
        if nsrc == 'same':
            neg_feats = X[np.logical_and(X[:, 1] == this_image_id,
                                     X[:, 2] != this_region_id)][:, 3:]
            if neg_feats.shape[0] == 0:
                print '  No neg samples from same image available.'
                print '  You should see this only rarely, otherwise better use method \'random\''
                continue
            randix = np.random.choice(range(len(neg_feats)), nneg)
        
            X_train.append(neg_feats[randix])
            y_train.extend([False] * len(randix))
            
        # Randomly sample negative examples from all regions
        #   not ever labelled with this word.
        if nsrc == 'random':
            sample_count = 0
            while sample_count < nneg:
                neg_sample = X[np.random.choice(len(X))]
                this_region_full_id = tuple(map(int, neg_sample[:3]))
                if this_region_full_id in wrd2dn[word]:
                    continue
                else:
                    X_train.append([neg_sample[3:]])
                    y_train.append(False)
                    sample_count += 1

    return np.concatenate(X_train, axis=0), y_train




def train_model(refdf, X, wordlist, classifier_spec, nneg=5, nsrc='same'):
    '''Train the WAC models, for wordlist.'''
    classifier, classf_params = classifier_spec
    wrd2dn = create_word2den(refdf, wordlist)
    clsf = {}
    for n, word in enumerate(wordlist):
        print('[%d/%d] training classifier for \'%s\'...'
                  % (n+1, len(wordlist), word))

        Xt, yt = make_train(X, wrd2dn, word, nneg, nsrc)
        npos = np.sum(yt)
        print(' (%d pos instances)' % (npos))
        this_classf = classifier(**classf_params)
        this_classf.fit(Xt,yt)
        clsf[word] = {'npos': npos,
                      'clsf': this_classf,
                      'nneg': nneg,
                      'nsrc': nsrc}
        print('... done.')
    return clsf