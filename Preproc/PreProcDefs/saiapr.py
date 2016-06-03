## SAIAPR

# Here I go via the original feature file. I only use this to get
# at the filenames and the region numbers..

featmat = scipy.io.loadmat('../Data/Images/SAIAPR/saiapr_features.mat')
X = featmat['X']

# get all the bounding boxes for SAIAPR regions
checked = {}
outrows = []
this_corpus = icorpus_code['saiapr']

for n, row in tqdm(enumerate(X)):
    this_image_id = int(row[0])
    this_region_id = int(row[1])
    this_category = int(row[-1])
    # Skip over b/w images. Test only once for each image.
    if checked.get(this_image_id) == 'skip':
        continue
    elif checked.get(this_image_id) != 'checked':
        img = plt.imread(saiapr_image_filename(this_image_id))
        checked[this_image_id] = 'checked'
        if len(img.shape) != 3:
            logging.info('skipping image %d' % (this_image_id))
            continue
    this_bb = get_saiapr_bb(this_image_id, this_region_id)
    if np.min(np.array(this_bb)) < 0:
        logging.info('skipping bb for %d %d' % (this_image_id, this_region_id))
        continue
    outrows.append((this_corpus, this_image_id, this_region_id, this_bb, this_category))

bbdf_saiapr = pd.DataFrame(outrows, columns='i_corpus image_id region_id bb cat'.split())

with gzip.open('PreProcOut/saiapr_bbdf.pklz', 'w') as f:
    pickle.dump(bbdf_saiapr, f)
