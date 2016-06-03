## Feature ablation: Only positional features

## 06, saia, only posit. feats

with gzip.open('../TrainModels/TrainedModels/model06_pos_s5r.pklz', 'r') as f:
    wac = pickle.load(f)

testdf = filter_by_filelist(srefdf, ssplit90['test'])

X_pos = np.concatenate([X[:,:3], X[:,-7:]], axis=1)

this_model = '06_pos_s5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X_pos, applyfunc=multiply_apply) ) )



## 07, saia, no posit. feats

with gzip.open('../TrainModels/TrainedModels/model07_nopos_s5r.pklz', 'r') as f:
    wac = pickle.load(f)

X_nopos = X[:,:-7]

this_model = '07_nopos_s5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X_nopos, applyfunc=multiply_apply) ) )


## 08, refcoco, only posit. feats

with gzip.open('../TrainModels/TrainedModels/model08_pos_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])

Xc_pos = np.concatenate([Xc[:,:3], Xc[:,-7:]], axis=1)

this_model = '08_pos_r5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc_pos, applyfunc=multiply_apply) ) )


## 09, refcoco, no posit. feats

with gzip.open('../TrainModels/TrainedModels/model09_nopos_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

Xc_nopos = Xc[:,:-7]

this_model = '09_nopos_r5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc_nopos, applyfunc=multiply_apply) ) )
