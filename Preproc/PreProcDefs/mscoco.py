## MSCOCO

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '... (MSCOCO)'
print now


with open('PreProcOut/refcoco_splits.json', 'r') as f:
    refcoco_splits = json.load(f)

with open('PreProcOut/google_refexp_rexsplits.json', 'r') as f:
    grex_splits = json.load(f)

all_coco_files = list(set(chain(*refcoco_splits.values())).union(set(chain(*grex_splits))))

coco_in_train_p = '../Data/Images/MSCOCO/annotations/instances_train2014.json'
with open(coco_in_train_p, 'r') as f:
    coco_in = json.load(f)

cocoandf = pd.DataFrame(coco_in['annotations'])

file_df = pd.DataFrame(all_coco_files, columns=['image_id'])

cocoandf_reduced = pd.merge(cocoandf, file_df)

bbdf_coco = cocoandf_reduced[['image_id', 'id', 'bbox', 'category_id']]

bbdf_coco['i_corpus'] = icorpus_code['mscoco']

bbdf_coco.columns = 'image_id region_id bb cat i_corpus'.split()

bbdf_coco = bbdf_coco['i_corpus image_id region_id bb cat'.split()]

with gzip.open('PreProcOut/mscoco_bbdf.pklz', 'w') as f:
    pickle.dump(bbdf_coco, f)
