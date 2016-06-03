## 02, but only top 20 classifiers

TOP = 20

with gzip.open('../TrainModels/TrainedModels/model01_s5r.pklz', 'r') as f:
    wac = pickle.load(f)

wac = dict(sorted(wac.items(), key=lambda x:x[1]['npos'])[::-1][:TOP])

testdf = filter_by_filelist(srefdf, ssplit90['test'])

this_model = '01_s5r; mult; top20'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=multiply_apply) ) )


## 03, but only top 20 classifiers

with gzip.open('../TrainModels/TrainedModels/model03_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

wac = dict(sorted(wac.items(), key=lambda x:x[1]['npos'])[::-1][:TOP])

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])

this_model = '03_r5r; mult; top20'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )


## 04, but only top 20 classifiers

with gzip.open('../TrainModels/TrainedModels/model04_g5r.pklz', 'r') as f:
    wac = pickle.load(f)

wac = dict(sorted(wac.items(), key=lambda x:x[1]['npos'])[::-1][:TOP])

testdf = filter_by_filelist(grefdf, rcocosplits['val'])

this_model = '04_g5r; mult; top20'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )



