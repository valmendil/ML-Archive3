#!/usr/bin/env python3

"""
Second (and top) layer of our approach. 
It uses the leaf model as it's first layer, combines it with predictions from Hough and feeds both results int a random forest classifier

"""

import matplotlib.image as mpimg
import numpy as np
import os

from sklearn import ensemble, preprocessing
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from skimage.transform import rotate

from utils import extract_img_list_patches, make_img_overlay, label_to_img, get_hot_labels, extract_neighborhoods, \
                        label_list_to_label_matrix, patch_abs_postition
                        
import leafModel
import houghUtils

""" --- FLAGS --- """

SAVE_L1_Overlay = True
RESTORE_LEAVE_MODEL = False

LABEL_PATCH_SIZE = 16

SEED = 66478

""" --- OTHER DECLARATIONS --- """
    
classifier_filename = '../../tmp/hough/houghModel_classifier'
prediction_mask_dir = "../../predictions/"
PATH_TO_PROBABILITY_ARRAY = "../../tmp/hough/colorProbability_array"
PATH_TO_FEATURE_BACKUP = "../../tmp/hough/"
    
# Size of the training images
TRAIN_IMG_SIZE = 400


class Model:
    
    def __init__(self, runNr):
        if not os.path.isdir(PATH_TO_FEATURE_BACKUP):
            os.mkdir(PATH_TO_FEATURE_BACKUP)
        #we need the entire image not just patches
        self.img_patch_size = TRAIN_IMG_SIZE
        self.img_patch_size_label = TRAIN_IMG_SIZE
        # model number used to distinguish different runs while doing cross validation.
        # the runNr is -1 for run.py and 1:... for cv
        self.runNr = runNr
        
        # setup estimator pipeline
        self.leave_model = leafModel.Model(runNr)
        clf = ensemble.RandomForestClassifier(n_estimators=2600, n_jobs=-2, random_state=SEED, verbose=3)
        scaler = preprocessing.StandardScaler()
        self.classifier = Pipeline([('scale', scaler), ('clf', clf)])

    def get_prediction(self, img, i):
        
        X = self.get_features(img, i)

        output_prediction = self.classifier.predict(X)
        
        return label_to_img(img.shape[0], img.shape[1], output_prediction, LABEL_PATCH_SIZE, LABEL_PATCH_SIZE)
    
    def train(self, train_imgs, gt_imgs):
        
        print("start training the first layer")
        print(" ----------------------------------------------------- ")
        if RESTORE_LEAVE_MODEL:
            self.leave_model.restoreModel()
        else:
            self.leave_model.train(train_imgs, gt_imgs, ENTIRE_SET=False)
            self.leave_model.saveModel()
        
        print(" ----------------------------------------------------- ")
            
        print("get predictions of the first layer")
        
        X = None
        for i in range(0, len(train_imgs)):
            img = train_imgs[i]

            features = self.get_features(img, i+1)            
            if X is None:
                X = features
            else:
                X = np.r_[X, features]
            
        X = np.asarray(X)

        print(" ----------------------------------------------------- ")    
        print("train second layer classifier")
        tl = extract_img_list_patches(gt_imgs, LABEL_PATCH_SIZE)
        labels = get_hot_labels(tl) #important in order to get one label per patch
        
        self.classifier.fit(X, labels)
        
        print("training completed")
        print(" ----------------------------------------------------- ")
        
        
      
    def saveModel(self):  
        #serialize classifier
        _ = joblib.dump(self.classifier, classifier_filename+str(self.runNr)+".joblib.pkl", compress=0)
        
        if not RESTORE_LEAVE_MODEL:
            self.leave_model.saveModel()
        
    def restoreModel(self):
        self.classifier = joblib.load(classifier_filename+str(self.runNr)+".joblib.pkl")          
        self.leave_model.restoreModel()
        
    def get_features(self, img, i):
        len_neib = 5
            
        def analyze(x):
            return [np.max(x), np.sum(x**2), np.mean(x), np.var(x)]
        
        def analyze_leave_model_prediction(output_prediction):
            print("process prediction of image "+str(i))
            neib = extract_neighborhoods([label_img], 1, len_neib)
             
            pos = len_neib
             
            feat_pred = []
            #compare the prediction with predictions of it's neighbors
            idx = 0
            for lp in neib: #neighborhood shape: (5,5)
                lp = np.asarray(lp)
                f = []
                 
                #horizontals
                f.append(analyze(lp[pos, :pos]))
                f.append(analyze(lp[pos, pos-3:pos]))
                f.append(analyze(lp[pos, pos-2:pos]))
                f.append(analyze(lp[pos, pos-1:pos]))
                f.append(analyze(lp[pos, pos+1:]))
                f.append(analyze(lp[pos, pos+1:pos+3+1]))
                f.append(analyze(lp[pos, pos+1:pos+2+1]))
                f.append(analyze(lp[pos, pos+1:pos+1+1]))
                 
                #verticals
                f.append(analyze(lp[:pos, pos]))
                f.append(analyze(lp[pos-3:pos, pos]))
                f.append(analyze(lp[pos-2:pos, pos]))
                f.append(analyze(lp[pos-1:pos, pos]))
                f.append(analyze(lp[pos:, pos]))
                f.append(analyze(lp[pos+1:pos+3+1, pos]))
                f.append(analyze(lp[pos+1:pos+2+1, pos]))
                f.append(analyze(lp[pos+1:pos+1+1, pos]))
                  
                #diagonals
                d = np.diag(lp)
                posd = int(np.ceil(len(d)/2))
                f.append(analyze(d[:posd]))
                f.append(analyze(d[posd-3:posd]))
                f.append(analyze(d[posd-2:posd]))
                f.append(analyze(d[posd-1:posd]))
                f.append(analyze(d[posd+1:]))
                f.append(analyze(d[posd+1:posd+3+1]))
                f.append(analyze(d[posd+1:posd+2+1]))
                f.append(analyze(d[posd+1:posd+1+1]))
                  
                d = np.diag(np.fliplr(lp))
                f.append(analyze(d[:posd]))
                f.append(analyze(d[posd-3:posd]))
                f.append(analyze(d[posd-2:posd]))
                f.append(analyze(d[posd-1:posd]))
                f.append(analyze(d[posd+1:]))
                f.append(analyze(d[posd+1:posd+3+1]))
                f.append(analyze(d[posd+1:posd+2+1]))
                f.append(analyze(d[posd+1:posd+1+1]))
                  
                #top, bottom, left, right area
                f.append(analyze(lp[:, pos:pos+2+1]))
                f.append(analyze(lp[:, pos-2:pos+1]))
                f.append(analyze(lp[pos:pos+2+1, :]))
                f.append(analyze(lp[pos-2:pos+1, :]))
                  
                #neighborhood
                f.append(analyze(lp))
                f.append(analyze(lp[pos-3:pos+3+1, pos-3:pos+3+1]))
                f.append(analyze(lp[pos-2:pos+2+1, pos-2:pos+2+1]))
                f.append(analyze(lp[pos-1:pos+1+1, pos-1:pos+1+1]))
                 
              
                f.append([lp[pos, pos]])
              
                #position - how close to border
                x, y = patch_abs_postition(label_img, 1, 1, idx)
                idx += 1
                f.append([ min(x-len_neib, 0, label_img.shape[0]-x-len_neib)]) 
                f.append([ min(y-len_neib, 0, label_img.shape[1]-y-len_neib)])
                #flatten
                f = [tuple_item for tuple_ in f for tuple_item in tuple_]
                feat_pred.append(f)    
                
            return feat_pred
            
        """ get and analyze predictions from the first layer for both the original and the rotated image (-45 degree) """    
            
        #get the prediction of the leaf model
        output_prediction, feat_c, feat_cp, feat_lbp, _ = self.leave_model.get_prediction(img, i, True)
        label_img = label_list_to_label_matrix(output_prediction, distinct=False)
        feat_pred_normal = analyze_leave_model_prediction(label_img)    
         
        #store predicted image
        if SAVE_L1_Overlay:
            pimg = label_to_img(img.shape[0], img.shape[1], output_prediction, LABEL_PATCH_SIZE, LABEL_PATCH_SIZE)
            oimg = make_img_overlay(img, pimg)
            oimg.save(prediction_mask_dir + "overlay_layer1pred_" + str(i) + ".png")  

 
        #get the prediction of the leaf model from the rotated image
        output_prediction_rot,_,_,_,_ = self.leave_model.get_prediction(rotate(img, -45), i,True, False)   
        label_img_rot = label_list_to_label_matrix(output_prediction_rot, distinct=False) 
        label_img_rot = rotate(label_img_rot, 45)
        feat_pred_rot = analyze_leave_model_prediction(label_img_rot)     
        
        #store prediction of the rotated image
        if SAVE_L1_Overlay:
            output_prediction_2 = extract_img_list_patches([label_img_rot], 1) # need back rotated labels
            pimg = label_to_img(img.shape[0], img.shape[1], output_prediction_2, LABEL_PATCH_SIZE, LABEL_PATCH_SIZE)
            oimg = make_img_overlay(img, pimg)
            oimg.save(prediction_mask_dir + "overlay_layer1pred_rot_" + str(i) + ".png")  
            
            
        print("generate hough transform of image "+str(i))
             
        hough_mask = houghUtils.draw_roads(
                                *houghUtils.find_roads(
                                   mpimg.imread(prediction_mask_dir +"mask_cp_%.3d" % i + ".png")))
                   
        mpimg.imsave(prediction_mask_dir +"mask_hough_%.3d" % i + ".png", hough_mask)
                           
        print("process hough predictions of image "+str(i))    
                          
        w = LABEL_PATCH_SIZE
        pos = int(np.ceil(w * 5 / 2))
        hough_patches = extract_neighborhoods([hough_mask], w, 2)
             
             
        feat_hough = []
        #analyze hough prediction image per neighborhood
        for lp in hough_patches: #neighborhood shape: (16*5,16*5)
            lp = np.asarray(lp)
            feature = []
                  
            #horizontals
            feature.append(analyze(lp[pos-8:pos+9, pos-2*w:pos]))
            feature.append(analyze(lp[pos-8:pos+9, pos-1*w:pos]))
            feature.append(analyze(lp[pos-8:pos+9, pos+1*w:]))
            feature.append(analyze(lp[pos-8:pos+9, pos+1*w:pos+(2+1)*w]))
            feature.append(analyze(lp[pos-8:pos+9, pos+1*w:pos+(1+1)*w]))
              
            #verticals
            feature.append(analyze(lp[:pos, pos-8:pos+9]))
            feature.append(analyze(lp[pos-2*w:pos, pos-8:pos+9]))
            feature.append(analyze(lp[pos-1*w:pos, pos-8:pos+9]))
            feature.append(analyze(lp[pos:, pos-8:pos+9]))
            feature.append(analyze(lp[pos+1*w:pos+(2+1)*w, pos-8:pos+9]))
            feature.append(analyze(lp[pos+1*w:pos+(1+1)*w, pos-8:pos+9]))
              
            #diagonals
            d = np.diag(lp)
            posd = int(np.ceil(len(d)/2))
            feature.append(analyze(d[:posd]))
            feature.append(analyze(d[posd-2*w:posd]))
            feature.append(analyze(d[posd-1*w:posd]))
            feature.append(analyze(d[posd+1*w:]))
            feature.append(analyze(d[posd+1*w:posd+(2+1)*w]))
            feature.append(analyze(d[posd+1*w:posd+(1+1)*w]))
              
            feature.append(analyze(np.diag(lp,-8)))
            feature.append(analyze(np.diag(lp,+8)))
              
            d = np.diag(np.fliplr(lp))
            feature.append(analyze(d[:posd]))
            feature.append(analyze(d[posd-2*w:posd]))
            feature.append(analyze(d[posd-1*w:posd]))
            feature.append(analyze(d[posd+1*w:]))
            feature.append(analyze(d[posd+1*w:posd+(2+1)*w]))
            feature.append(analyze(d[posd+1*w:posd+(1+1)*w]))
              
            feature.append(analyze(np.diag(lp,-8)))
            feature.append(analyze(np.diag(lp,+8)))
              
            #top, bottom, left, right area
            feature.append(analyze(lp[:, pos:pos+(2+1)*w]))
            feature.append(analyze(lp[:, pos-2*w:pos+1*w]))
            feature.append(analyze(lp[pos:pos+(2+1)*w, :]))
            feature.append(analyze(lp[pos-2*w:pos+1*w, :]))
              
            #neighborhood
            feature.append(analyze(lp))
            feature.append(analyze(lp[pos:pos+w, pos:pos+w]))
            feature.append(analyze(lp[pos-1*w:pos+(1+1)*w, pos-1*w:pos+(1+1)*w]))
          
            #flatten
            feature = [tuple_item for tuple_ in feature for tuple_item in tuple_]
            feat_hough.append(feature)        
            
                 
        #stack features together    
        feat_pred_normal = np.asarray(feat_pred_normal)
        feat_pred_rot = np.asarray(feat_pred_rot)
        feat_hough = np.asarray(feat_hough)
        return np.c_[feat_pred_normal, feat_pred_rot, feat_hough, feat_c, feat_cp, feat_lbp]