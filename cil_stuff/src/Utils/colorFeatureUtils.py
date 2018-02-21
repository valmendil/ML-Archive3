#!/usr/bin/env python3

"""
    this module contains functions that allow to extract some widely used features such as histograms and image moments
"""

import numpy as np


# Extract a histogram and 6-dimensional features consisting of average RGB color as well as variance and the local energy
def extract_color_features(img_patch):
    feat_m = np.mean(img_patch, axis=(0,1))
    feat_v = np.var(img_patch, axis=(0,1))
    
    # Local Energy
    feat_e = np.sum(img_patch**2, axis=(0,1))
    
    # histogramm
    feat_hist2, _ = np.histogram(img_patch, density=True)
    
    feat = np.concatenate((feat_m, feat_v , feat_e,  feat_hist2))
    return feat

# Extract features of gray color images such as average as well as variance, a histogram and some moments
def extract_moments_and_histogram(image):
    feat_m = [np.mean(image)]
    feat_v = [np.var(image)]
    feat_max = [np.max(image)]
    
    # Local Energy
    feat_e = [np.sum(image**2)]
    
    # histogramm
    feat_hist, _ = np.histogram(image, density=True)
    
    #moments
    x, y = np.mgrid[:image.shape[0],:image.shape[1]]
    moments = {}
    with np.errstate(invalid='ignore'):
        moments['mean_x'] = np.sum(x*image)/np.sum(image)
        moments['mean_y'] = np.sum(y*image)/np.sum(image)
                  
        # raw or spatial moments
        moments['m00'] = np.sum(image)
        moments['m01'] = np.sum(x*image)
        moments['m10'] = np.sum(y*image)
        moments['m11'] = np.sum(y*x*image)
        moments['m02'] = np.sum(x**2*image)
        moments['m20'] = np.sum(y**2*image)
        moments['m12'] = np.sum(x*y**2*image)
        moments['m21'] = np.sum(x**2*y*image)
        moments['m03'] = np.sum(x**3*image)
        moments['m30'] = np.sum(y**3*image)
        
        moments['mu11'] = np.sum((x-moments['mean_x'])*(y-moments['mean_y'])*image)
        moments['mu02'] = np.sum((y-moments['mean_y'])**2*image) # variance
        moments['mu20'] = np.sum((x-moments['mean_x'])**2*image) # variance
        moments['mu12'] = np.sum((x-moments['mean_x'])*(y-moments['mean_y'])**2*image)
        moments['mu21'] = np.sum((x-moments['mean_x'])**2*(y-moments['mean_y'])*image) 
        moments['mu03'] = np.sum((y-moments['mean_y'])**3*image) 
        moments['mu30'] = np.sum((x-moments['mean_x'])**3*image) 
              
        # central standardized or normalized or scale invariant moments
        moments['nu11'] = moments['mu11'] / np.sum(image)**(2/2+1)
        moments['nu12'] = moments['mu12'] / np.sum(image)**(3/2+1)
        moments['nu21'] = moments['mu21'] / np.sum(image)**(3/2+1)
        moments['nu20'] = moments['mu20'] / np.sum(image)**(2/2+1)
        moments['nu03'] = moments['mu03'] / np.sum(image)**(3/2+1) # skewness
        moments['nu30'] = moments['mu30'] / np.sum(image)**(3/2+1) # skewness
      
    moments = np.asarray(list(moments.values()))
    moments[np.isnan(moments)] = 0
    moments[np.isinf(moments)] = 0
  
    feat = np.concatenate((feat_m, feat_v , feat_max, feat_e,  feat_hist, moments))
    return feat

