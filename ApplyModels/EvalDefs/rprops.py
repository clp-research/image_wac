## model 01 (s5r) on Berkeley edgebox proposals

with gzip.open('../TrainModels/TrainedModels/model01_s5r.pklz', 'r') as f:
    wac = pickle.load(f)

propfilelist = list(set(b_bbdf['image_id'].tolist()))

testdf_props = filter_by_filelist(srefdf, propfilelist)

testdf_props['gbb'] = get_gold_bbs(testdf_props, s_bbdf)

rprp_df = get_rprp_bbs(propfilelist, b_bbdf)

testdf_props['ious'] = testdf_props.apply(lambda x: apply_iou_to_refdf_row(x, rprp_df), axis=1)

this_model = '01_s5r; brprop; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf_props, wac, Xb, applyfunc=multiply_apply) ) )


## model 05 (sr5r) on Berkeley edgebox proposals

with gzip.open('../TrainModels/TrainedModels/model05_sr5r.pklz', 'r') as f:
    wac = pickle.load(f)

this_model = '05_sr5r; brprop; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf_props, wac, Xb, applyfunc=multiply_apply) ) )




## model 03 (r5r) on google rprops

with gzip.open('../TrainModels/TrainedModels/model03_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

testdf = filter_by_filelist(rrefdf, rcocosplits['val'])

propfilelist = list(set(g_bbdf['image_id'].tolist()))

testdf_props = filter_by_filelist(rrefdf, propfilelist)

testdf_props['gbb'] = get_gold_bbs(testdf_props, c_bbdf)

rprp_df = get_rprp_bbs(propfilelist, g_bbdf)

result_ious = []
for n, row in testdf_props.iterrows():
    result_ious.append(apply_iou_to_refdf_row(row, rprp_df))

testdf_props['ious'] = result_ious


this_model = '03_r5r; grprop; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf_props, wac, Xg, applyfunc=multiply_apply) ) )



## Ablations

## 09 r5r, no positional features; on google rprops

with gzip.open('../TrainModels/TrainedModels/model09_nopos_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

Xg_np = Xg[:,:-7]

this_model = '09_nopos_r5r; grprop; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf_props, wac, Xg_np, applyfunc=multiply_apply) ) )



## 08 r5r, only pos; on google rprops

with gzip.open('../TrainModels/TrainedModels/model08_pos_r5r.pklz', 'r') as f:
    wac = pickle.load(f)

Xg_pos = np.concatenate([Xg[:,:3], Xg[:,-7:]], axis=1)

this_model = '08_pos_r5r; grprop; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf_props, wac, Xg_pos, applyfunc=multiply_apply) ) )

