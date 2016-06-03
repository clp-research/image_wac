# coding: utf-8

'''
Some utility functions for the image WAC project
'''
from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
from PIL import Image as PImage


SAIAPR_BASEDIR = '../Data/Images/SAIAPR/saiapr_tc-12'
MSCOCO_BASEDIR = '../Data/Images/MSCOCO/train2014'
IMAGENET_TEMPLATE = '../Data/Images/ImageNet/SelectedImages/%s/%s.JPEG'
#IMAGENET_TEMPLATE = '../Exploration/ImageNet/%s/%s.JPEG'


icorpus_code = {
    'saiapr': 0,   # the original SAIAPR corpus; original regions
    'mscoco': 1,    # MSCOCO; original bounding boxes
    'saiapr_berkeley': 2, # SAIAPR, with berkeley region proposals
    'mscoco_grprops': 3, # MSCOCO, region proposals as per google refexp
    'image_net': 4 # ImageNet; with bbs
}

code_icorpus = {item:key for key,item in icorpus_code.items()}


def plot_labelled_bb(impath, bblist, title=None, text_size='large',
                     mode='path', omode='screen', opath = None):
    '''Given the path of an image and a list containing tuples 
    of bounding box (a list of x,y,w,h) and label (str), 
    plot these boxes and labels into the image.
    If mode is path, impath is path, if it is image, impath is the actual
    image on top of which the bbs are to be drawn.
    If omode is screen, assumption is that matplotlib functions are
    displayed directly (i.e., IPython inline mode), if it is
    img, function saves the plot to opath.'''

    if mode=='path':
        img = plt.imread(impath)
    elif mode=='img':
        img = impath

    fig, ax = plt.subplots()
    fig.set_size_inches(10,20)
    ax.imshow(img)

    if bblist is not None:
        for (this_bb, this_label) in bblist:
            x,y,w,h = this_bb

            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor='r',
                    fill=False      # remove background
                )
            )
            if this_label != '':
                ax.text(x,y, this_label, size=text_size, style='italic',
                        bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    if title is not None:
        ax.set_title(title)
    ax.axis('off')

    if omode == 'img':
        fig.savefig(opath, bbox_inches = 'tight', pad_inches = 0)
        plt.close()  # to supress showing the plot in interactive mode


def saiapr_basepath(image_id):
    '''return the basepath for an SAIAPR image, given the image ID'''
    if len(str(image_id)) == 5:
        directory = str(image_id)[:2]
    elif len(str(image_id)) == 4:
        directory = '0' + str(image_id)[0]
    else:
        directory = '00'
    return directory


def saiapr_image_filename(image_id):
    '''return the path of a SAIAPR image, given the image ID'''
    directory = saiapr_basepath(image_id)
    return SAIAPR_BASEDIR + '/' + directory + '/images/' + str(image_id) + '.jpg'

def saiapr_mask_filename(image_id, region_id):
    '''return the path of a SAIAPR mask, given the image ID and region ID'''
    directory = saiapr_basepath(image_id)
    return SAIAPR_BASEDIR + '/' + directory + '/segmentation_masks/' + \
                   str(image_id) + '_' + str(region_id) + '.mat'


def get_saiapr_bb(image_id, region_id):
    '''get the bounding box of an SAIAPR region, given image and region IDs'''
    mask_path = saiapr_mask_filename(image_id, region_id)
    #print mask_path
    mask = scipy.io.loadmat(mask_path)
    mask = mask['segimg_t']
    mask = mask + 1
    x1, y1 = np.nonzero(mask)[1].min(), np.nonzero(mask)[0].min()
    x2, y2 = np.nonzero(mask)[1].max(), np.nonzero(mask)[0].max()
    return [x1,y1,x2-x1,y2-y1]


def mscoco_image_filename(image_id):
    '''get the image path for an MSCOCO image (from train2014), 
       given the ID'''
    return os.path.join(MSCOCO_BASEDIR, 'COCO_train2014_%012d.jpg' % (image_id))


def join_imagenet_id(image_id, region_id):
    return 'n%08d_%d' % (image_id, region_id)

def get_imagenet_filename(image_id):
    folder = image_id.split('_')[0]
    return IMAGENET_TEMPLATE  % (folder, image_id)


def get_image_filename(icorp, image_id):
    if 'mscoco' in code_icorpus[icorp]:
        return mscoco_image_filename(image_id)
    if 'saiapr' in code_icorpus[icorp]:
        return saiapr_image_filename(image_id)
    if 'image_net' in code_icorpus[icorp]:
        return get_imagenet_filename(image_id)
    raise ValueError('Unknown corpus code')



# 2016-03-31 moved here from extract_feats.py

def get_thumbnail((old_image_id, img), i_corpus, image_id, bb, 
                  resize=True,
                  xs=224,ys=224):
    if old_image_id != image_id:
        this_path = get_image_filename(i_corpus, image_id)
        # this_icorpus = code_icorpus[i_corpus]
        # if 'saiapr' in this_icorpus:
        #     this_path = saiapr_image_filename(image_id)
        # elif 'mscoco' in this_icorpus:
        #     this_path = mscoco_image_filename(image_id)
        # else:
        #     #logging.warn('unknown corpus?? Skipping. (%s)' % (this_icorpus))
        #     print 'unknown corpus?? Skipping. (%s)' % (this_icorpus)
        #     return -1
        img = plt.imread(this_path)
    #else:
    #    print 'saved the planet one i/o access!!'

    # need to clip bounding box to 0, because the google region
    #   weirdly sometimes have negative coordinates (?!):
    #print bb, img.shape
    x,y,w,h = np.clip(np.array(bb), 0, np.max(img.shape))
    w = img.shape[1]-x if x+w >= img.shape[1] else w
    h = img.shape[0]-y if y+h >= img.shape[0] else h
    #print 'after', x,y,w,h, 

    img_cropped = img[y:y+h,x:x+w]
    if resize:
        pim = PImage.fromarray(img_cropped)
        pim2 = pim.resize((xs,ys), PImage.ANTIALIAS)
        img_resized = np.array(pim2)
    else:
        img_resized = img_cropped
    return ((image_id, img), img_resized)






def filter_by_filelist(refdf, filelist):
    return pd.merge(refdf, pd.DataFrame(filelist, columns=['image_id']))


def filter_X_by_filelist(X, filelist):
    tmp_df = pd.DataFrame(X)
    return np.array(tmp_df[tmp_df.iloc[:,1].isin(filelist)])


# separation plot
# Author: Cameron Davidson-Pilon,2013
# see http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf

def separation_plot( p, y, title=None, **kwargs ):
    """
    This function creates a separation plot for logistic and probit classification. 
    See http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf
    
    p: The proportions/probabilities, can be a nxM matrix which represents M models.
    y: the 0-1 response variables.
    
    """    
    assert p.shape[0] == y.shape[0], "p.shape[0] != y.shape[0]"
    n = p.shape[0]

    try:
        M = p.shape[1]
    except:
        p = p.reshape( n, 1 )
        M = p.shape[1]

    #colors = np.array( ["#fdf2db", "#e44a32"] )
    colors_bmh = np.array( ["#eeeeee", "#348ABD"] )


    fig = plt.figure( )#figsize = (8, 1.3*M) )
    
    for i in range(M):
        ax = fig.add_subplot(M, 1, i+1)
        ix = np.argsort( p[:,i] )
        #plot the different bars
        bars = ax.bar( np.arange(n), np.ones(n), width=1.,
                color = colors_bmh[ y[ix].astype(int) ], 
                edgecolor = 'none')
        ax.plot( np.arange(n+1), np.append(p[ix,i], p[ix,i][-1]), "k",
                 linewidth = 1.,drawstyle="steps-post" )
        #create expected value bar.
        ax.vlines( [(1-p[ix,i]).sum()], [0], [1] )
        #ax.grid(False)
        #ax.axis('off')
        plt.xlim( 0, n)
        
    if title is not None:
        plt.title(title)
        
    plt.tight_layout()
    
    return
