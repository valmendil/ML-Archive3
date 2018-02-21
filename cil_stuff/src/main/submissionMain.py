#!/usr/bin/env python3

"""
This code produces the submission file for the CIL project on road segmentation.
run.py calls the function make_submission which translates the predictions into a 
list of labels and stores it into a file

"""
from utils import make_img_overlay, make_img_savable
from mask_to_submission import masks_to_submission

import matplotlib.image as mpimg
from PIL import Image
import os
import shutil

import NNModel
import exampleSKLModel
import colorProbabilityModel
import leafModel
import deepModel

""" 
    models 
    choose the model by setting the MODEL_ID 
    possible id's:
        1 -> NNModel                 (cnn baseline)
        2 -> exampleSKLModel         (simple baseline using mean and variance as features and logistic regression as estimator)
        3 -> colorProbabilityModel   (tries to predict the color probability cp )
        4 -> leafModel               (first layer of our approach)
        5 -> deepModel               (contains the code of the second layer and runs both layers of our approach)
        
""" 

MODEL_ID = 5


""" flags """ # change them as much as you like

SUBMISSION_FILENAME = 'dummy_submission.csv'

""" directories """ # don't change them
test_path = '../../data/test_set_images/'
prediction_dir = '../../predictions/'
submission_dir = '../../submissions/'




def make_submission(model, submission_filename):  

    """ --- FOLDERS --- """    
    #delete old prediction folder and make a new one
    if os.path.isdir(prediction_dir):
        shutil.rmtree(prediction_dir)    
    os.mkdir(prediction_dir)
    if not os.path.isdir(submission_dir):
        os.mkdir(submission_dir)

    """ SUBMISSION """
    image_filenames = []
    for i in range(1, 51):
        prediction_filename = prediction_dir + "prediction_" + '%.3d' % i + ".png"
        image_filenames.append(prediction_filename)
        #get test data
        imageid = 'test_' + str(i) +'/test_' + str(i)
        image_filename = test_path + imageid + ".png"
        img = mpimg.imread(image_filename)
        #make predictions and save them as overlay images        
        pimg = model.get_prediction(img, i)
        Image.fromarray(make_img_savable(pimg)).save(prediction_filename)
        oimg = make_img_overlay(img, pimg)
        oimg.save(prediction_dir + "overlay_" + '%.3d' % i + ".png") 
    #create submission file  
    masks_to_submission(submission_dir + submission_filename, *image_filenames)
    print('submission stored in directory '+submission_dir+submission_filename)


if __name__ == '__main__':
    runNr = -1 #indicator if we run cross validation or try to get a submission (-1 stands for submission)
    if MODEL_ID == 1:
        model = NNModel.Model(runNr)
    elif MODEL_ID == 2:
        model = exampleSKLModel.Model(runNr)
    elif MODEL_ID == 3:
        model = colorProbabilityModel.Model(runNr)  
    elif MODEL_ID == 4:
        model = leafModel.Model(runNr)   
    else:
        model = deepModel.Model(runNr)
        
    model.restoreModel()
    make_submission(model, SUBMISSION_FILENAME)