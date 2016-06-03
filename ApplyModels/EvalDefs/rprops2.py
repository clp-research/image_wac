## model 04 (g5r) on google rprops

with gzip.open('../TrainModels/TrainedModels/model04_g5r.pklz', 'r') as f:
    wac = pickle.load(f)

propfilelist = list(set(g_bbdf['image_id'].tolist()))

# save, since we only have features for testset anyway
testdf_props = filter_by_filelist(grefdf, propfilelist)

testdf_props['gbb'] = get_gold_bbs(testdf_props, c_bbdf)

rprp_df = get_rprp_bbs(propfilelist, g_bbdf)

result_ious = []
for n, row in testdf_props.iterrows():
    result_ious.append(apply_iou_to_refdf_row(row, rprp_df))

testdf_props['ious'] = result_ious


this_model = '04_g5r; grprop; mult'

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model

results.append((this_model,
                eval_testdf(testdf_props, wac, Xg, applyfunc=multiply_apply) ) )

