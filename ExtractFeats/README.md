# Extracting the image features

Now we need feature representations for the regions. We represent the images using the final fully connected layer (1024 dimensions; before softmax) of a large convolutional neural network that was trained in an object recognition task. Note that while the representations it learned were optimized for that set of objects, in themselves they are not specific to them.

We augment these features with a small set (7) of features that provide information about the position of the region within the image (relative x,y coordinates, distance to center, relative area).

## Background: The image features

The particular network we use was described by

> Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, A. Rabinovich. Going Deeper with Convolutions, arXiv:1409.4842, September 2014.

and named by them "GoogLeNet". We use it at some steps removed from the source, though: We use the [sklearn-theano](http://sklearn-theano.github.io/feature_extraction/index.html#feature-extraction) port of the [Caffe replication](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) of this network structure. This network was trained on the data from the [Large Scale Visual Recognition Challenge 2014 (ILSVRC2014)](http://www.image-net.org/challenges/LSVRC/2014/), i.e., on ImageNet data.


## Steps

Just run

```
python extract_feats.py
```

and take a nice walk, or watch a movie. Or two. (It takes about 5 hours to extract features for all 364k regions on our i7/16 GB/GTX 970 machine.)
