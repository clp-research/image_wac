## 05, s+r, on s

with gzip.open('../TrainModels/TrainedModels/model05_sr5r.pklz', 'r') as f:
    wac = pickle.load(f)

testdf = filter_by_filelist(srefdf, ssplit90['test'])

this_model = '05_sr5r->s; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=multiply_apply) ) )


## 05, s+r, on r

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])

this_model = '05_sr5r->r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )


## 05, s+r, on g

testdf = filter_by_filelist(grefdf, rcocosplits['val'])

this_model = '05_sr5r->g; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=multiply_apply) ) )



