## 03, refcoco

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])

with gzip.open('../TrainModels/TrainedModels/model03_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

this_model = '03_r5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )



## 03, refcoco, on grex

testdf = filter_by_filelist(grefdf, rcocosplits['val'])

with gzip.open('../TrainModels/TrainedModels/model03_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

this_model = '03_r5r->gr; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )


