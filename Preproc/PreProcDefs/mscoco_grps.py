## MSCOCO, Google Region Proposals

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '... (MSCOCO, Google RProps)'
print now

with open('../Data/RefExps/MSCOCO/google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json', 'r') as f:
    grex_json = json.load(f)

gimdf = pd.DataFrame(grex_json['images']).T

refcoco_testfiledf = pd.DataFrame(list(chain(refcoco_splits['testA'], 
                                             refcoco_splits['testB'], 
                                             refcoco_splits['val'])), 
                                  columns=['image_id'])

gimdf_reduced = pd.merge(gimdf, refcoco_testfiledf)

rows = []
this_i_corpus = icorpus_code['mscoco_grprops']
for n, row in tqdm(gimdf_reduced.iterrows()):
    bbs = row['region_candidates']
    this_image_id = row['image_id']
    for k, this_bbs in enumerate(bbs):
        this_bb = this_bbs['bounding_box']
        this_cat = this_bbs['predicted_object_name']
        rows.append([this_i_corpus, this_image_id, k, this_bb, this_cat])


bbdf_cocorprop = pd.DataFrame(rows, 
                              columns='i_corpus image_id region_id bb cat'.split())


with gzip.open('PreProcOut/cocogrprops_bbdf.pklz', 'w') as f:
    pickle.dump(bbdf_cocorprop, f)


