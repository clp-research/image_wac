## 01

this_model = '01_s5r; add'

with gzip.open('../TrainModels/TrainedModels/model01_s5r.pklz', 'r') as f:
    wac = pickle.load(f)

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

testdf = filter_by_filelist(srefdf, ssplit90['test'])

results.append((this_model, eval_testdf(testdf, wac, X)))


this_model = '01_s5r; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=multiply_apply) ) )



this_model = '01_s5r; hmean'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=hmean_apply) ) )

