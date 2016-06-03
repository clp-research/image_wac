## Baseline, max area, SAIAPR

testdf = filter_by_filelist(srefdf, ssplit90['test'])

this_model = 'max area; s'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, X, applyfunc=apply_areabaseline_to_image) ) )


## Baseline, max area, COCO

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])

this_model = 'max area; c'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf, wac, Xc, applyfunc=apply_areabaseline_to_image) ) )


