#!/usr/bin/env python3

"""
    this model tries to estimate the probability of a pixel being street based on it's color value
    the same code in a adapted form also appears in the leaf model
"""

import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from utils import make_img_overlay, img_float_to_uint8, label_to_img, extract_img_patches, get_hot_labels


""" --- FLAGS --- """

BIT_DEPTH = 8
PIXEL_DEPTH = 255

IMG_SIZE = 400 



""" --- OTHER DECLARATIONS --- """

prediction_mask_dir = "../../predictions/"

PATH_TO_PROBABILITY_ARRAY = "../../tmp/colorProbability_array.npy"
if not os.path.isdir('../../tmp/'):
    os.mkdir('../../tmp/')



def pos(x, y, z, BIT_DEPTH):
    return (x>>(8-BIT_DEPTH))*(2**BIT_DEPTH)*(2**BIT_DEPTH)+(y>>(8-BIT_DEPTH))*(2**BIT_DEPTH)+(z>>(8-BIT_DEPTH))


class Model:
    
    def __init__(self, runNr):
        #we require the entire image not just it's patches
        self.img_patch_size = IMG_SIZE
        self.img_patch_size_label = IMG_SIZE 
    
    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(self, filename):
        img = mpimg.imread(filename)
        img_prediction = self.get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)
        return oimg

    def get_prediction(self, img, idx):
#         img_patches = extract_img_patches(img, self.img_patch_size)
        print("predict img "+str(idx))
        
        test = img_float_to_uint8(img)
        pred_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        im = np.zeros((img.shape[0], img.shape[1]))
    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                im[i, j] = self.road_prob[pos(test[i,j,0], test[i,j,1], test[i,j,2], BIT_DEPTH)]
                pred_mask[i, j]=im[i, j]*PIXEL_DEPTH


        Image.fromarray(pred_mask).save(prediction_mask_dir +"mask_pre_%.3d" % idx + ".png")
                
        output_prediction = get_hot_labels(extract_img_patches(pred_mask / PIXEL_DEPTH, 16))
        return label_to_img(img.shape[0], img.shape[1], output_prediction, 16, 16)
    
    def train(self, imgs, gt_imgs):
        color_freq = np.ones(((2**BIT_DEPTH)**3))
        is_road = np.zeros(((2**BIT_DEPTH)**3))

        for im in range(len(imgs)):
            img = img_float_to_uint8(imgs[im])
            grt = img_float_to_uint8(gt_imgs[im])
            print("Processing image ", im)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    color_freq[pos(img[i,j,0], img[i,j,1], img[i,j,2], BIT_DEPTH)] += 1
                    is_road[pos(img[i,j,0], img[i,j,1], img[i,j,2], BIT_DEPTH)] += grt[i,j]/PIXEL_DEPTH
                
        print("All pixels processed:", np.sum(color_freq))
        print("Road pixels found:", np.sum(is_road))
    
        self.road_prob = is_road/color_freq
        self.road_prob[color_freq == 1] = 0 # assign 0 for all unseen colors, may be changed in future
      
    def saveModel(self):  
        np.save(PATH_TO_PROBABILITY_ARRAY, self.road_prob)
        print("Probabity array saved in ", PATH_TO_PROBABILITY_ARRAY)
        
    def restoreModel(self):
        print("Loading", PATH_TO_PROBABILITY_ARRAY)
        self.road_prob = np.load(PATH_TO_PROBABILITY_ARRAY)
        
    
    