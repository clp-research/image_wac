## SAIAPR, Berkeley EdgeBox proposals

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '... (SAIAPR, Berkeley EdgeBox)'
print now

with open('../Preproc/PreProcOut/saiapr_90-10_splits.json', 'r') as f:
    saiapr90_10 = json.load(f)

outrows = []
this_corpus = icorpus_code['saiapr_berkeley']
for this_image_id in tqdm(saiapr90_10['test'][:BERKELEYMAX]):
    this_bbs = get_berkeley_edgeboxes(this_image_id)
    for n, bb in enumerate(this_bbs):
        outrows.append((this_corpus, this_image_id, n, list(bb.astype(int)), -1))


bbdf_berkeley =  pd.DataFrame(outrows, columns='i_corpus image_id region_id bb cat'.split())

with gzip.open('PreProcOut/berkeley_bbdf.pklz', 'w') as f:
    pickle.dump(bbdf_berkeley, f)
