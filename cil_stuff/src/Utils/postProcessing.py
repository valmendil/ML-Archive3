#!/usr/bin/env python3

"""
    this module contains some functions for some simple form of postprocessing
"""

import numpy as np
import scipy.ndimage as ndimage

PATCH_SIZE = 16
FOREGROUND_THRESHOLD = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

def _patch_to_label(patch):
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def _fix_patches(values):
    edges = values[1]+values[3]+values[5]+values[7]
    me = values[4]
    corners = values[0]+values[2]+values[6]+values[8]
    if me==1 and edges <=1 and corners<=1:
        return 0
    if me==0 and edges>=3 and corners>=3:
        return 1
    else:
        return me


def postprocess_image(img, rounds=1):
    """
    Performs postprocessing of mask images.

    Parameters:
        img -- patched binary mask image
        rounds -- how many times postprocessing will be applied

    The parameters are exactly what is returned by find_roads (see there).

    Returns:
        Postprocessed binary mask image.

    """

    res_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    arr = np.zeros((img.shape[0]//PATCH_SIZE, img.shape[1]//PATCH_SIZE), dtype=np.uint8)

    for i in range(0, img.shape[0], PATCH_SIZE):
        for j in range(0, img.shape[1], PATCH_SIZE):
            patch = img[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            arr[i//PATCH_SIZE, j//PATCH_SIZE] = _patch_to_label(patch)

    for _ in range(rounds):
        arr = ndimage.generic_filter(arr, _fix_patches, footprint=np.ones((3, 3)))

    for i in range(0, img.shape[0], PATCH_SIZE):
        for j in range(0, img.shape[1], PATCH_SIZE):
            res_img[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = arr[i//PATCH_SIZE, j//PATCH_SIZE]*255

    return res_img

