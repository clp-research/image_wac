# Getting The Data #

The work reported here is based on a number of public datasets (some of them quite large), which need to be available to reproduce our results. This document explains how to get them. The data must be linked to from the directory that contains this document. The intended directory structure is shown at the end of this document.

## Image Corpora

### "Segmented and Annotated IAPR TC-12 dataset"

This dataset contains 20k images, mostly holiday-type snaps (outdoor scenes). 

The images themselves are from the "IAPR TC-12" dataset ( <http://www.imageclef.org/photodata>), described in

Grubinger, Clough, Müller, & Deselaers. (2006). The IAPR TC-12 benchmark-a new evaluation resource for visual information systems. International Conference on Language Resources and Evaluation., 13–23.

From this paper: "The majority of the images are provided by viventura, an independent travel company that organizes adventure and language trips to South-America. At least one travel guide accompanies each tour and they maintain a daily online diary to record the adventures and places visited by the tourists (including at least one corresponding photo). Furthermore, the guides provide general photographs of each location, accommodation facilities and ongoing social projects."

We use an augmented distribution of this imageset that also contains region segmentations (hence "Segmented and Annotated IAPR TC-12 dataset").

The dataset is described here: <http://imageclef.org/SIAPRdata>.

But the link to the actual data on that page was dead when I tried it in June 2015; I got the data directly from the first author of the following publication (which gives a further description of the dataset).

[1] Hugo Jair Escalante, Carlos A. Hernández, Jesus A. Gonzalez, A. López-López, Manuel Montes, Eduardo F. Morales, L. Enrique Sucar, Luis Villaseñor and Michael Grubinger.  The Segmented and Annotated IAPR TC-12 Benchmark. Computer Vision and Image Understanding, doi: <http://dx.doi.org/10.1016/j.cviu.2009.03.008>, in press, 2009. 

The directory `saiapr_tc-12` from this distribution needs to be accessible at `Images/SAIAPR/saiapr_tc-12` from the directory of this document.

We also need the `features.mat` from `matlab/`, linked to `saiapr_features.mat`.


#### Automatically generated region proposals for SAIAPR

The following paper also uses this dataset:

Ronghang Hu, Huazhe Xu, Marcus Rohrbach, Jiashi Feng, Kate Saenko, Trevor Darrell. Natural Language Object Retrieval. <http://arxiv.org/abs/1511.04164>. Nov 2015.

(Project page: <http://ronghanghu.com/object-retrieval/>.)

As part of their work, they computed region proposals for all images. (Using EdgeBox, and taking the 100 top proposals per image.) To compare against their results, we use these as well. We got them from the first author of that paper.

The data needs to be in `Images/SAIAPR/Berkeley_rprops`. This directory also contains the split into train/test that the authors used.


### "Microsoft Common Objects in Context"

We also use the MS COCO dataset. This is a (much larger) dataset of pictures containing certain objects in a wide variety of settings.

We use the `train2014` set, available via the [MSCOCO download page](http://mscoco.org/dataset/#download) or directly by executing the following command:

```bash
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
```

Attention! This is a rather large set (13GB, 80k images).

This directory must be available from `Images/MSCOCO`.

We also need some of the annotations that are part of the MSCOCO dataset, namely the bounding boxes of the pre-segmented objects in the images. These go in `Images/MSCOCO/annotations` (i.e., `Images/MSCOCO/annotations/instances_train2014.json`).



## Referring Expressions

The second ingredient are human-produced expressions referring to objects in these images.



### ReferIt/SAIAPR

Using the [ReferIt game](http://tamaraberg.com/referitgame/), Tamara Berg and colleagues collected expressions referring to the regions from the SAIAPR data. The work is described here:

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, Tamara L. Berg.   ReferItGame: Referring to Objects in Photographs of Natural Scenes. Empirical Methods in Natural Language Processing (EMNLP) 2014.  Doha, Qatar.  October 2014. 

The resulting (120k) referring expressions are available from:
<http://tamaraberg.com/referitgame/ReferitData.zip>

They need to be placed at `RefExps/SAIAPR/ReferIt`.


### ReferIt/MSCOCO

Using the same setup, the same group also collected references to objects in the MSCOCO data. We kindly received this data from them via email.

This data needs to be at `RefExps/MSCOCO/ReferIt_COCO`.



### Google_Refexp

Finally, the following paper also described a data collection of what they call `unambiguous object descriptions', using the COCO data. (Note that these descriptions aren't quite called "referring expressions".)

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, Kevin Murphy. Generation and Comprehension of Unambiguous Object Descriptions. <http://arxiv.org/abs/1511.02283>.

Project page: <https://github.com/mjhucla/Google_Refexp_toolbox>

The `json` that we used needs to be computed out of the MSCOCO annotations and additional data that they provide, by following the instructions at their github page. When following these instructions, a directory `google_refexp_dataset_release` is created. This needs to be accessible at `RefExps/MSCOCO/google_refexp_dataset_release`.



# Summary: The Intended Layout

```bash
026b_image_wac % tree -L 4 -l --filelimit 10 Data/
Data/
|-- GETDATA.md
|-- Images
|   |-- MSCOCO
|   |   |-- annotations
|   |   |   `-- instances_train2014.json
|   |   `-- train2014 [82783 entries exceeds filelimit, not opening dir]
|   `-- SAIAPR
|       |-- Berkeley_rprops
|       |   |-- README.txt
|       |   |-- edgeboxes
|       |   |-- referit_test_imlist.txt
|       |   `-- referit_trainval_imlist.txt
|       |-- saiapr_features.mat
|       `-- saiapr_tc-12 [39 entries exceeds filelimit, not opening dir]
`-- RefExps
    |-- MSCOCO
    |   |-- ReferIt_COCO
    |   |   |-- README.txt
    |   |   |-- cleaned(google).p
    |   |   |-- cleaned(licheng).p
    |   |   `-- instances.json
    |   `-- google_refexp_dataset_release
    |       |-- README.md
    |       |-- google_refexp_train_201511.json
    |       |-- google_refexp_train_201511_coco_aligned.json
    |       |-- google_refexp_val_201511.json
    |       |-- google_refexp_val_201511_coco_aligned.json
    |       |-- license.txt
    |       `-- spatial_phrases.txt
    `-- SAIAPR
        `-- ReferIt
            |-- CannedGames.txt
            |-- README.txt
            |-- RealGames.txt
            `-- test_set_ground_truth
```


