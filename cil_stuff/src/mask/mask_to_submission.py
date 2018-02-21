#!/usr/bin/env python3

"""
    convert a list of labels / predictions into a submission
"""

import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

submission_filename = 'dummy_submission.csv'

""" directories """ # don't change them
submission_dir = '../../submissions/'
predictions_filename_prefix = '../../predictions/prediction_' 
#predictions_filename_prefix = '../../data/training/groundtruth/satImage_' 

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def mask_to_submission_lists(image_filenames):
    """Reads a list of images and outputs the numbers that should go into the submission file"""
    patch_size = 16
    labels = []
    for image_filename in image_filenames:      
        im = mpimg.imread(image_filename)
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                labels.append(label)
    return labels 

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    image_filenames = []
    for i in range(1, 51):
        image_filename = predictions_filename_prefix + '%.3d' % i + ".png"
        print ( image_filename )
        if not os.path.isdir(submission_dir):
            os.mkdir(submission_dir)
        image_filenames.append(image_filename)
    masks_to_submission(submission_dir + submission_filename, *image_filenames)
