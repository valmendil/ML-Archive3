#!/usr/bin/env python3

"""
simple model. It takes the mean and the variance as features and feeds them into a logistic regression model.
The main code was taken from the cil tutorial

In the result section of the report we refer to this model as LogReg model (B)

"""

import numpy as np
import os

from utils import extract_img_patches, label_to_img, get_hot_labels
from sklearn import linear_model
from sklearn.externals import joblib

""" --- FLAGS --- """

GET_TPR = True
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16


""" --- OTHER DECLARATIONS --- """

classifier_filename = '../../tmp/exampleSKLModel_classifier.joblib.pkl'
if not os.path.isdir('../../tmp/'):
    os.mkdir('../../tmp/')

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img_patch):
    feat_m = np.mean(img_patch, axis=(0,1))
    feat_v = np.var(img_patch, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img_patch):
    feat_m = np.mean(img_patch)
    feat_v = np.var(img_patch)
    feat = np.append(feat_m, feat_v)
    return feat


class Model:
    
    def __init__(self, runNr):
        self.img_patch_size = IMG_PATCH_SIZE
        self.img_patch_size_label = IMG_PATCH_SIZE
        self.classifier = linear_model.LogisticRegression(C=1e5, class_weight="balanced")    
   
    def get_prediction(self, img, i):
        img_patches = extract_img_patches(img, self.img_patch_size)
        
        X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
        output_prediction = self.classifier.predict(X)
        
        return label_to_img(img.shape[0], img.shape[1], output_prediction, self.img_patch_size, self.img_patch_size)
    
    def train(self, img_patches, train_labels):
        Y = get_hot_labels(train_labels) #important in order to get one label per patch
        
        X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
        self.classifier.fit(X, Y)
        
        if GET_TPR:
            # Predict on the training set
            Z = self.classifier.predict(X)
            
            # Get non-zeros in prediction and grountruth arrays
            Zn = np.nonzero(Z)[0]
            Yn = np.nonzero(Y)[0]
            
            TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
            print('True positive rate = ' + str(TPR))

      
    def saveModel(self):  
        #serialize classifier
        _ = joblib.dump(self.classifier, classifier_filename, compress=9)
        
    def restoreModel(self):
        self.classifier = joblib.load(classifier_filename)   
        
    
    