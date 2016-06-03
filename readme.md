# Resolving References to Objects in Photographs

This repository contains the code required to reproduce the results reported in

* David Schlangen, Sina Zarrieß, and Casey Kennington. 2016. Resolving References to Objects in Photographs using the Words-As-Classifiers Model. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)*, Berlin, Germany. [[pdf](Papers/ACL2016/schlangen_etal_acl2016.pdf)]

Doing so requires taking several steps:

1. First, you need to collect the data, and place it where the scripts expect it. This is described in the [readme in `Data/`](Data/README.md).
2. Since we use a collection of corpora, we first preprocess them to represent the information in a uniform format. This is described in [`Preproc`](Preproc/README.md).
2. Then you need to extract the features from the images; for this you need to download the weights of the convolutional network that we've used, and have various dependencies installed. (Or, if you want to run your own experiments, you can apply whatever method you want for turning the image regions into a feature vector.) This is described in [`ExtractFeats`](ExtractFeats/README.md).
3. Now it's time to train the words-as-classifiers models. See [`TrainModels`](TrainModels/README.md).
4. How well did we do? Run the evaluations in [`ApplyModels`](ApplyModels/readme.md).
5. Now you can reproduce the tables and figures from the paper. See [TablesAndFigures.ipynb](Papers/ACL2016/TablesAndFigures.ipynb).

While some care has been taken not to hard code too many things, it is not unlikely that you will have to changes some paths.

Questions? Queries? Comments. Email us! (*firstname.lastname*@uni-bielefeld.de)


**Acknowledgements**: Big thanks are due to the groups that shared the fruits of their hard work collecting images, referring expressions and region markup. [Hu et al. "Natural Language Object Retrieval"](http://ronghanghu.com/text_obj_retrieval/), [Mao et al. "Generation and comprehension of unambiguous object descriptions"](https://github.com/mjhucla/Google_Refexp_toolbox), and Tamara Berg's [ReferItGame collection](http://tamaraberg.com/referitgame/) ([MSCOCO data here](https://github.com/lichengunc/refer)). More references [here](Data/README.md).

This work was supported by the Cluster of Excellence "Cognitive Interaction Technology" (CITEC; EXC 277) at Bielefeld University, funded by the German Research Foundation (DFG).

*David Schlangen, 2016-06-01*
