#!/usr/bin/python

"""
    transform a submission into a set of prediction images
"""

import os
from PIL import Image
import math
import numpy as np
import shutil

label_file = 'dummy_submission.csv'

h = 16
w = h
imgwidth = int(math.ceil((600.0/w))*w)
imgheight = int(math.ceil((600.0/h))*h)
nc = 3

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open('../../submissions/'+label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    
    #delete old prediction folder
    prediction_dir = "../../predictions/"
    if os.path.isdir(prediction_dir):
        shutil.rmtree(prediction_dir)    
    os.mkdir(prediction_dir)  
    
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

        Image.fromarray(im).save('../../predictions/prediction_' + '%.3d' % image_id + '.png')

    return im

for i in range(1, 5):
    reconstruct_from_labels(i)
   
