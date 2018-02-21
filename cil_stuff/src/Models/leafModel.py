#!/usr/bin/env python3

"""
The model of the first layer of our approach. The results are feed into the second layer and further processed.
The model consists of data augmentation and a gradient boosting model that uses HOG, LBP and our color probability (cp) as Features
trainingMain and run.py can run the code as an individual model that is independent from the second layer too

"""

from skimage.feature import hog
from skimage import color
import numpy as np
from PIL import Image

from sklearn import  ensemble, preprocessing
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern
from skimage.transform import rotate

from colorFeatureUtils import extract_moments_and_histogram, extract_color_features
from utils import extract_img_list_patches, label_to_img, get_hot_labels, extract_neighborhoods, img_float_to_uint8


""" --- FLAGS --- """

PATCH_SIZE = 16
NEIGHBORHOOD_SIZE = 2

lbp_radius = 3

BIT_DEPTH = 8
PIXEL_DEPTH = 255
SEED = 66478

""" --- OTHER DECLARATIONS --- """
    
classifier_filename = '../../tmp/hough/houghModel_classifier'
prediction_mask_dir = "../../predictions/"
PATH_TO_PROBABILITY_ARRAY = "../../tmp/hough/colorProbability_array"
    
# size of a training image
TRAIN_IMG_SIZE = 400


def pos(x, y, z, BIT_DEPTH):
    return (x>>(8-BIT_DEPTH))*(2**BIT_DEPTH)*(2**BIT_DEPTH)+(y>>(8-BIT_DEPTH))*(2**BIT_DEPTH)+(z>>(8-BIT_DEPTH))

def extract_hog_features(img_neighborhoods):
    feat_hog = []
    for n in img_neighborhoods:
        fd = hog(n, orientations=8, pixels_per_cell=(PATCH_SIZE, PATCH_SIZE),
                cells_per_block=(1, 1), visualise=False)
        feat_hog.append(fd)
    return np.matrix(feat_hog)   
    

def extract_lbp_features(img_neighborhoods):
    feat_lbp = []
    n_bins = 26
    for n in img_neighborhoods:
        lbp = local_binary_pattern(n, 8*lbp_radius, lbp_radius, 'uniform')   
        #large neighborhood
        hist3, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins)
        #surrounding patches
        hist2, _ = np.histogram(lbp[PATCH_SIZE*(NEIGHBORHOOD_SIZE-1):PATCH_SIZE*(NEIGHBORHOOD_SIZE+2), PATCH_SIZE*(NEIGHBORHOOD_SIZE-1):PATCH_SIZE*(NEIGHBORHOOD_SIZE+2) ].ravel(), 
                                        density=True, bins=n_bins)
        #patch
        hist1, _ = np.histogram(lbp[PATCH_SIZE*(NEIGHBORHOOD_SIZE):PATCH_SIZE*(NEIGHBORHOOD_SIZE+1), PATCH_SIZE*(NEIGHBORHOOD_SIZE):PATCH_SIZE*(NEIGHBORHOOD_SIZE+1) ].ravel(), 
                                        density=True, bins=n_bins)
        feat_lbp.append(np.concatenate((hist3, hist2, hist1))) 
    return np.matrix(feat_lbp)     
            


class Model:
    
    def __init__(self, runNr):
        # the leaf model needs the entire image as an input not only patches like the other models do 
        self.img_patch_size = TRAIN_IMG_SIZE
        self.img_patch_size_label = TRAIN_IMG_SIZE
        self.runNr = runNr
        
        # setup estimator pipeline
        clf = ensemble.GradientBoostingClassifier(n_estimators=1100, learning_rate=0.1, subsample = 0.5, random_state=SEED, verbose=2)
        scaler = preprocessing.StandardScaler()
        self.classifier = Pipeline([('scale', scaler), ('clf', clf)])
        
    def get_cp_features(self, img, itId, SAVE_MASK=True):
        test = img_float_to_uint8(img)
        col_pred_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        im = np.zeros((img.shape[0], img.shape[1]))
        
        # get cp preditions per image
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                im[i, j] = self.road_prob[pos(test[i,j,0], test[i,j,1], test[i,j,2], BIT_DEPTH)]
                col_pred_mask[i, j]=im[i, j]*255
   
        # crop output images of cp into patches
        col_pred_patches = extract_img_list_patches([col_pred_mask], PATCH_SIZE)
        
        #save output of cp
        if SAVE_MASK:
            Image.fromarray(col_pred_mask).save(prediction_mask_dir +"mask_cp_%.3d" % itId + ".png")     

        return np.asarray([ extract_moments_and_histogram(col_pred_patches[i])
                                    for i in range(col_pred_patches.shape[0])])   
    

    def get_prediction(self, img, idx, get_raw_prediction=False, SAVE_MASK=True):
        
        print("leaf_model: predict img "+str(idx))
        
        #get all the patches
        img_patches = extract_img_list_patches([img], PATCH_SIZE)
        
        #get all the neighborhoods around each patch of the grey scale image
        img_neighborhoods = extract_neighborhoods([color.rgb2gray(img)], PATCH_SIZE, NEIGHBORHOOD_SIZE)
        
        """ Histogram and Moments """
        feat_c = np.asarray([ extract_color_features(img_patches[i]) for i in range(len(img_patches))])
        X = feat_c
                
        """ COLOR PREDICTION """
        feat_cp = self.get_cp_features(img, idx, SAVE_MASK)
        X = np.c_[X, feat_cp]
            
        """ HOG """
        feat_hog = extract_hog_features(img_neighborhoods)
        X = np.c_[X, feat_hog]   
            
        """ LBP """
        feat_lbp = extract_lbp_features(img_neighborhoods)
        X = np.c_[X, feat_lbp] 
            
        #train classifier
        output_prediction = self.classifier.predict(X)
        
        if get_raw_prediction:
            # if used by the second layer: return features and the prediction
            return (output_prediction, feat_c, feat_cp, feat_lbp, feat_hog)
        
        else:
            # if used as an individual module by trainingMain: return the prediction as an image
            return label_to_img(img.shape[0], img.shape[1], output_prediction, PATCH_SIZE, PATCH_SIZE)
    
    def train(self, all_train_imgs, all_gt_imgs, ENTIRE_SET=True):
        total_len = len(all_train_imgs)
        
        if not ENTIRE_SET:
            # only train on a subset of the training set. 
            # if the prediction of the first layer is too good in comparison with hough the second layer doesn't yield good results
            lenSet = int(np.ceil(total_len * 0.6))
            print("train leave layer only on "+str(lenSet)+" images of "+str(total_len))
            train_imgs = all_train_imgs[0: lenSet]
            gt_imgs = all_gt_imgs[0: lenSet]
        else:
            # train on all the training data that we have
            train_imgs = all_train_imgs
            gt_imgs = all_gt_imgs
            
        len_train_l1 = len(train_imgs)

        # Data Augmentation: artificially increase the number of training data by changing orientation and brightness
        # helps against over fitting
        l = int(np.ceil(len_train_l1 / 4 ))
        for i in range(max(0,lenSet-l), total_len):
            train_imgs = np.append(train_imgs, [rotate(all_train_imgs[i],-45)], axis=0)
            gt_imgs = np.append(gt_imgs, [rotate(all_gt_imgs[i], -45)], axis=0)
        for i in range(0, l*2):
            train_imgs = np.append(train_imgs, [rotate(train_imgs[i],-175)*0.9], axis=0)
            gt_imgs = np.append(gt_imgs, [rotate(gt_imgs[i], -175)], axis=0)
        for i in range(l*2 , l*3):
            train_imgs = np.append(train_imgs, [rotate(train_imgs[i], 20)*0.95], axis=0)
            gt_imgs = np.append(gt_imgs, [rotate(gt_imgs[i], 20)], axis=0)
        for i in range(l , l*2):
            train_imgs = np.append(train_imgs, [rotate(train_imgs[i], -20)], axis=0)
            gt_imgs = np.append(gt_imgs, [rotate(gt_imgs[i], -20)], axis=0)
        for i in range(l*3 , len_train_l1):
            train_imgs = np.append(train_imgs, [rotate(train_imgs[i], 55)*0.80+10], axis=0)
            gt_imgs = np.append(gt_imgs, [rotate(gt_imgs[i], 55)], axis=0)
        for i in range(0, len_train_l1):
            train_imgs = np.append(train_imgs, [rotate(train_imgs[i], 87)], axis=0)
            gt_imgs = np.append(gt_imgs, [rotate(gt_imgs[i], 87)], axis=0)
            
                
        
        print("preprocess")
        img_patches = extract_img_list_patches(train_imgs, PATCH_SIZE)
        
        tl = extract_img_list_patches(gt_imgs, PATCH_SIZE)
        labels = get_hot_labels(tl) #important in order to get one label per patch
        
        """  Histogram and Moments  """
        print("get color features")
        feat_c = np.asarray([ extract_color_features(img_patches[i]) for i in range(len(img_patches))])
        
        """ COLOR PREDICTION """
        def train_cp(start, end, color_freq, is_road):
            for im in range(0, min(40, len_train_l1)):
                img = img_float_to_uint8(all_train_imgs[im])
                grt = img_float_to_uint8(all_gt_imgs[im])
                print("Processing image ", im)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        color_freq[pos(img[i,j,0], img[i,j,1], img[i,j,2], BIT_DEPTH)] += 1
                        is_road[pos(img[i,j,0], img[i,j,1], img[i,j,2], BIT_DEPTH)] += grt[i,j]/PIXEL_DEPTH
                    
            print("All pixels processed:", np.sum(color_freq))
            print("Road pixels found:", np.sum(is_road))
        
            with np.errstate(invalid='ignore'):
                road_prob = is_road/color_freq
            mask = np.isnan(road_prob)
            road_prob[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), road_prob[~mask])
            return (road_prob, color_freq, is_road)
        
        
        print("train color prediction model")
        color_freq = np.ones(((2**BIT_DEPTH)**3))
        is_road = np.zeros(((2**BIT_DEPTH)**3))
        self.road_prob, color_freq, is_road = train_cp(0, min(20, len_train_l1), color_freq, is_road)
         
        print("get color prediction features ")
        feat_cp = self.get_cp_features(train_imgs[0], 0)
        for ii in range(1, len(train_imgs)):
            feat_cp = np.r_[feat_cp, self.get_cp_features(train_imgs[ii], ii, False)]
             
        print("train cp on the remaining training images too and update the road probability")   
        self.road_prob, _ , _ = train_cp(min(20, len_train_l1), total_len, color_freq, is_road)


        print(feat_cp.shape)
            
        #get all the neighborhoods around each patch of all the grey scale images    
        gray_imgs = []
        for gim in train_imgs:
            gray_imgs.append(color.rgb2gray(gim))
        img_neighborhoods = extract_neighborhoods(gray_imgs, PATCH_SIZE, NEIGHBORHOOD_SIZE)
                
                
        """ HOG """
        print("get HOG")
        feat_hog = extract_hog_features(img_neighborhoods)
        print(feat_hog.shape)
            
            
        """ LBP """
        print("get LBP")
        feat_lbp = extract_lbp_features(img_neighborhoods)
        print(feat_lbp.shape)
        
        
        print("train classifier")
        
        # stack all the features together
        X = feat_c
        X = np.c_[X, feat_cp]
        X = np.c_[X, feat_hog]
        X = np.c_[X, feat_lbp]
        
        self.classifier.fit(X, labels)
        
        print("training completed")
        
        
      
    def saveModel(self):  
        #serialize classifier
        _ = joblib.dump(self.classifier, classifier_filename+str(self.runNr)+".joblib.pkl", compress=0)
        
        np.save(PATH_TO_PROBABILITY_ARRAY+str(self.runNr)+".npy", self.road_prob)
        print("Probabity array saved in ", PATH_TO_PROBABILITY_ARRAY+str(self.runNr)+".npy")
        
    def restoreModel(self):
        self.classifier = joblib.load(classifier_filename+str(self.runNr)+".joblib.pkl")  
        
        print("Loading", PATH_TO_PROBABILITY_ARRAY+str(self.runNr)+".npy")
        self.road_prob = np.load(PATH_TO_PROBABILITY_ARRAY+str(self.runNr)+".npy") 