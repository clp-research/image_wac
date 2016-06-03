# Preprocessing the data

This directory contains the scripts for preprocessing the data from `../Data`.
The referring expressions and image region proposals all came in somewhat different formats. These are converted into a uniform data structure which will form the basis for the further processing.


## Compiling the referring expressions

First, execute

```
python preproc_refexps.py
```


This creates one `refdf` (a Pandas DataFrame) for each of the refexp corpora, as well as `json` files for the splits. It uses the perceptron tagger from `nltk` to POS tag the referring expressions.

The resulting tables have the following columns:

```
i_corpus | image_id	| region_id | r_corpus | rex_id | refexp | tagged
```

This script will also precompile directories that hold the various suggested splits, as well as our own 90/10 split on SAIAPR.

(Warning: This can take a while. On my 2013 i7 Macbook Air, it takes about 12 minutes to create these files and write them to disk.)


## Compiling the bounding boxes of the image regions of interest

Then, execute

```
python preproc_region_defs.py
```

This creates `bbdf`s, DataFrames containing bounding boxes, indexed by the same columns as used in the `refdf`s. These tables have the following columns:

```
i_corpus | image_id	| region_id | bb | cat
```

Where `bb` is the bounding box (a list of `x`, `y`, `w`, `h`), and `cat` is the category of this region, if in the original file there was one give. (For the google rprops, it's the predicted category of the proposal.)

(This will take an even longer while. On my MBA, it takes about 26 minutes. Most of that is spent in i/o, computing the SAIAPR regions, because for each of those, a file must be read in.)

All together, this will compute 363,489 regions in 4 DataFrames.


When all this is completed, you should have the following:

```
Preproc % tree PreProcOut/
PreProcOut/
|-- berkeley_bbdf.pklz
|-- cocogrprops_bbdf.pklz
|-- google_refexp_rexsplits.json
|-- grex_refdf.pklz
|-- mscoco_bbdf.pklz
|-- refcoco_refdf.pklz
|-- refcoco_splits.json
|-- saiapr_90-10_splits.json
|-- saiapr_bbdf.pklz
|-- saiapr_berkeley_10-10_splits.json
`-- saiapr_refdf.pklz
```



