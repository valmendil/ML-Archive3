"""
    Collection of important helper functions. The functions allow to extract data from files, to split images in patches
    and neighborhoods, to extract labels, to rebuild images from labels, etc.
    
"""

import os
import matplotlib.image as mpimg
from PIL import Image
import numpy
from numpy import zeros
import math


PIXEL_DEPTH = 255
NUM_CHANNELS = 3 # RGB images
NUM_LABELS = 2

VERBOSE_EXTRACTING = False

def img_crop(im, w, h):
    """ split image into patches """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_img_list_patches(imgs, PATCH_SIZE): 
    num_images = len(imgs)
    img_patches = [img_crop(imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return numpy.asarray(data)

def extract_img_patches(img, PATCH_SIZE): 
    #there is actually a sklearn function too that does something similar (extract_patches_2d) !!!
    return numpy.asarray(img_crop(img, PATCH_SIZE, PATCH_SIZE))

def extract_data(filename, img_ids, PATCH_SIZE):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in img_ids:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if VERBOSE_EXTRACTING:
                print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)

    img_patches = [img_crop(imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)

def img_crop_neighborhoods(im, w, h, nr_additional_patches):
    """
    wrap an image with a white boundary and extract a list of all neighborhoods
    that surround a patch in the given image im
    w, h are the width and the height of a patch
    the number of additional patches stands for how many patches are added on each side of a patch 
    """
    is_2d = len(im.shape) < 3
    list_patches = []
    im = numpy.asarray(im)
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    add_p_w = nr_additional_patches * w
    add_p_h = nr_additional_patches * h
    
    #wrap the image with a 0 boundary
    if is_2d:
        wraped_img = zeros((imgwidth+add_p_w*2,imgheight+add_p_h*2))
        wraped_img[add_p_w:add_p_w+imgwidth, add_p_h:add_p_h+imgheight] = im
    else:
        wraped_img = zeros((imgwidth+add_p_w*2,imgheight+add_p_h*2, 3))
        wraped_img[add_p_w:add_p_w+imgwidth, add_p_h:add_p_h+imgheight, :] = im
        
    # extract
    for i in range(0,imgheight, h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = wraped_img[j:j+w+add_p_w*2, i:i+h+add_p_h*2]
            else:
                im_patch = wraped_img[j:j+w+add_p_w*2, i:i+h+add_p_h*2, :]
            list_patches.append(im_patch)
    return list_patches

def extract_neighborhoods(imgs, PATCH_SIZE, nr_additional_patches):
    """
        extract a list of neighborhoods for each patch in all the images
        the number of additional patches stands for how many patches are added on each side of a patch 
    """
    data = []
    for i in range(len(imgs)):
        img_patches = img_crop_neighborhoods(imgs[i], PATCH_SIZE, PATCH_SIZE, nr_additional_patches)
        for j in range(len(img_patches)):
            data.append(img_patches[j])
    return data

   

def value_to_class(v, threshold=0.25):
    """
         Assign a label to a patch v
    """
    foreground_threshold = threshold # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]   #positive
    else:
        return [1, 0]   #negative

# Extract label images
def extract_labels(filename, img_ids, PATCH_SIZE):
    """
        Extract the labels into a 1-hot matrix [image index, label index].
    """
    gt_imgs = []
    for i in img_ids:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if VERBOSE_EXTRACTING:
                print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(num_images)]
    return numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    
def get_hot_labels(data, threshold=0.25):   
    """ extract one label per patch """
    labels = numpy.asarray([value_to_class(numpy.mean(data[i]), threshold)[1] for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def get_hot_labels_2d(data):   
    """ extract one label per patch """
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

def label_to_img(imgwidth, imgheight, labels, patch_w, patch_h):
    """ Convert array of labels to an image """
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,patch_h):
        for j in range(0,imgwidth,patch_w):
            if labels[idx] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+patch_w, i:i+patch_h] = l
            idx = idx + 1
    return array_labels

def make_img_savable(gt_img):
    """ convert 2d prediction to an image with  3 channels """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels != 3:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        gt_img = gt_img_3c
    return gt_img

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    with numpy.errstate(invalid='ignore'):
        rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
        
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def label_list_to_label_matrix(labels, distinct=True):
    """ 
    create a matrix that contains the labels for all the patches in the same order as the original image. 
    the result is similar to a subsampled version of label_to_image 
    """
    w = int(numpy.ceil(math.sqrt(len(labels))))
    h = w
    labelMatrix = numpy.zeros([w, h])
    idx = 0
    for i in range(0,h):
        for j in range(0,w):
            if distinct:
                if labels[idx] > 0.5:
                    l = 1
                else:
                    l = 0
            else:
                l = labels[idx]
            labelMatrix[j, i] = l
            idx = idx + 1
    return labelMatrix

def subsample_to_label_matrix(im, w, h, smooth=True):
    """ 
    create a matrix that contains the labels for all the patches. the result is similar to a subsampled version of label_to_image 
    It's required that the image is a prediction or a ground truth image
    """
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    labelMatrix = numpy.zeros([numpy.ceil(imgwidth / w), numpy.ceil(imgheight / h)])
    
    is_2d = len(im.shape) < 3
    ii = 0
    for i in range(0,imgheight,h):
        jj = 0
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
                
            l = numpy.mean(im_patch)
            if smooth: 
                l = value_to_class(l)[1]
            labelMatrix[jj, ii] = l   
            jj +=1
        ii +=1
    print(ii)
                
    return labelMatrix

def patch_abs_postition(im, w, h, idx): 
    """
        get the position in the underlying image for a label with a certain index number 
        in the label list. the index idx starts with 0    
    """
    imgwidth = im.shape[0]
    len_w =numpy.ceil(imgwidth/w)
    x = numpy.floor(idx / len_w) 
    y = (idx % len_w) 
    return (x,y)
    
def patch_rel_postition(im, w, h, idx):    
    """
        get the relative position in the underlying image for a patch with a certain index number 
        values get larger (up to 255) the closer the patch gets to a boundary
    """
    len_w = numpy.ceil(im.shape[0] / w)
    len_h = numpy.ceil(im.shape[1] / h)
    x, y = patch_abs_postition(im, w, h, idx)
    x = numpy.exp(2.83321334406*(2*numpy.abs((x-len_h/2))/len_h))*15
    y = numpy.exp(2.83321334406*(2*numpy.abs((y-len_w/2))/len_w))*15
    return (x,y)

