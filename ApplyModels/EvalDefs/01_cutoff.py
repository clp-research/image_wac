## 01

this_model = '01_co75_s5r; mult'

with gzip.open('../TrainModels/TrainedModels/model01_s5r.pklz', 'r') as f:
    wac = pickle.load(f)

print strftime("%Y-%m-%d %H:%M:%S")
print 'Now evaluating:', this_model



testfiles = ssplit90['test']


# compute accuracy scores of classifiers on first half of testset



def score_wac_on_data(refdf, X, wac, word, nneg=3, nsrc='random'):
    w2d = create_word2den(refdf, [word])
    if word not in w2d:
        return np.nan
    X_t, y_t = make_train(X, w2d, word, nneg, nsrc)
    res = wac[word]['clsf'].score(X_t, y_t)
    return res




splitpoint = int(len(testfiles) / 2)

conffiles = testfiles[:splitpoint]
confdf = filter_by_filelist(srefdf, conffiles)

confdf['is_rel'] = confdf['refexp'].apply(is_relational)
confdf_norel = confdf.query('is_rel == False')

X_val = X[np.array([True if int(image_id) in conffiles else False for image_id in X[:,1]])]

for word in wac.keys():
    wac[word]['conf'] = score_wac_on_data(confdf_norel, X_val, wac, word)


# evaluate on second half of testset

testfiles = testfiles[splitpoint:]
testdf = filter_by_filelist(srefdf, testfiles)

results.append((this_model,
                eval_testdf(testdf, wac, X,
                            applyfunc=multiply_apply, 
                            cutoff=0.75) ) )

