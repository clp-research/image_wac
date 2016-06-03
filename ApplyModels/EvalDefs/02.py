## 02 

with gzip.open('../TrainModels/TrainedModels/model02_s5s.pklz', 'r') as f:
    wac = pickle.load(f)

this_model = '02_s5s; add'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model, eval_testdf(testdf, wac, X)))


this_model = '02_s5s; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=multiply_apply) ) )


this_model = '02_s5s; hmean'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=hmean_apply) ) )


