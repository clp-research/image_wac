

## 04, gre

testdf = filter_by_filelist(grefdf, rcocosplits['val'])
with gzip.open('../TrainModels/TrainedModels/model04_g5r.pklz', 'r') as f:
    wac = pickle.load(f)

this_model = '04_g5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )


## 04, grex, on refcoco

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])
with gzip.open('../TrainModels/TrainedModels/model04_g5r.pklz', 'r') as f:
    wac = pickle.load(f)

this_model = '04_g5r->rc; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )


