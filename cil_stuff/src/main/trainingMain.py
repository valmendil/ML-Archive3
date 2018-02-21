#!/usr/bin/env python3

"""
Training Methods and Cross Validation
choose the Model by importing the correct model Module! Set the flags in order to specify the size of the training and validation set!
The trained Model gets saved and can be reused later on.

"""
from utils import extract_data, extract_labels, make_img_overlay, make_img_savable
from mask_to_submission import mask_to_submission_lists
import os
import shutil
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import f1_score
        
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

""" flags """ 
    # change them as much as you like
    # only used locally !!! run.py is not influenced by this definitions
    
    
TRAIN_MODEL = False # independent of cv. True if the model is meant to be reused later on
TRAINING_SIZE = 100 # max = 100


ESTIMATE_F1_SCORE = True
CV_TRAINING_SIZE = 75 # max = 100
CV_VALIDATION_SIZE = 25 # Size of the validation set. max = 100 - Training_size ; min = 0
MAX_K = 1 # maximal number of runs of the cross validation. min = 1


modelNr = -1 #standard

""" directories """ # don't change them
data_dir = '../../data/training/'
prediction_dir = "../../predictions/"


def train_classifier(model, img_ids, saveModel=True):  

    """ --- DATA EXTRACTION --- """
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, img_ids, model.img_patch_size)
    train_labels = extract_labels(train_labels_filename, img_ids, model.img_patch_size_label)

    """ --- Train Classifier --- """
    model.train(train_data, train_labels)
    
    if saveModel:
        model.saveModel()
    

def validate(model, img_ids, cleanup_prediction_folder=True):  
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 
    
    #delete old prediction folder
    if os.path.isdir(prediction_dir) and cleanup_prediction_folder:
        shutil.rmtree('../../predictions')    
    if not os.path.isdir(prediction_dir):
        os.mkdir(prediction_dir)
    
    #get Labels from ground truth and don't reuse train_labels since the patch size might be changed
    img_prediction_filenames =[]
    img_groundtruth_filenames = []
    for i in img_ids:
        imageid = "satImage_%.3d" % i
        filename = train_data_filename + imageid + ".png"
        img = mpimg.imread(filename)
        #prediction
        pimg = model.get_prediction(img, i)
        img_prediction_filename = prediction_dir + "prediction_" + '%.3d' % i + ".png"
        img_prediction_filenames.append(img_prediction_filename)
        Image.fromarray(make_img_savable(pimg)).save(img_prediction_filename)
        #overlay image with the prediction
        oimg = make_img_overlay(img, pimg)
        oimg.save(prediction_dir + "overlay_" + str(i) + ".png")  
        #get ground truth file
        gt_filename = train_labels_filename + imageid + ".png"
        img_groundtruth_filenames.append(gt_filename)
    
    #F1 score
    predictions = mask_to_submission_lists(img_prediction_filenames)
    labels = mask_to_submission_lists(img_groundtruth_filenames)
    return f1_score(labels, predictions, average='binary')


def doCV(modelInit, training_size, validation_size, maxK=100, saveModel = False):
    
    """ --- CHECK VALIDITY OF ARGUMENTS --- """
    if training_size + validation_size > 100:
        print('validation_size + training_size should be smaller than 101')
        return
    if training_size > 100:
        print('TRAINING_SIZE should be less than 100')  
        return
    if training_size < 1:
        print('VALIDATION_SIZE should be bigger than 1')  
        return    
    print ("Run Cross Validation on a training set of size "+str(training_size)+" with a validation set size of "+str(validation_size))
    
    """ --- RUN CROSS VALIDATION --- """
    total_size = training_size + validation_size
    valset_start_id = 1
    all_training_ids = list(range(1, total_size+1))
    score = 0
    run = 1
    while valset_start_id < total_size and run <= maxK:
        print("-----------------------------------------------")
        print(" run: "+str(run))
        print("-----------------------------------------------")
        model = modelInit(run)
        #define validation and training set for each run
        valset_end_id = valset_start_id+validation_size
        val_set = range(valset_start_id, valset_end_id)
        training_set = []
        if valset_start_id > 1:
            training_set += all_training_ids[:valset_start_id-1]
        if valset_end_id < total_size:
            training_set += all_training_ids[valset_end_id:]
        #run
        train_classifier(model, training_set, saveModel) # (model isn't stored in tmp; only locally)
        new_score = validate(model, val_set, valset_start_id==1) # calculates F1 Score of the valset using the trained model
        print("score of run "+str(run)+": "+str(new_score))
        #average
        score = score + ( new_score - score) / run
        run += 1
        valset_start_id = valset_end_id
    
    return score




if __name__ == '__main__':
    
    #choose the correct Model
    if MODEL_ID == 1:
        Model = NNModel.Model
    elif MODEL_ID == 2:
        Model = exampleSKLModel.Model
    elif MODEL_ID == 3:
        Model = colorProbabilityModel.Model 
    elif MODEL_ID == 4:
        Model = leafModel.Model   
    else:
        Model = deepModel.Model
    
    
    if not os.path.isdir('../../tmp/'):
        os.mkdir('../../tmp/')
    if not os.path.isdir('../../predictions/'):
        os.mkdir('../../predictions/')
    
    if ESTIMATE_F1_SCORE:
        print("run cross validation")
        score = doCV(Model, CV_TRAINING_SIZE, CV_VALIDATION_SIZE, MAX_K)
        
        print("------------------------------------")
        print('Resulting F1 Scores: ' + str(score)  )    
        print("------------------------------------")
    
    model = Model(modelNr)
    if TRAIN_MODEL:
        print("Train a Model with a training set of size "+str(TRAINING_SIZE))
        train_classifier(model, range(0, TRAINING_SIZE+1))
        
    print('run completed')
    

    

       
    
    