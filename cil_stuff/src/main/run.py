#!/usr/bin/env python3

"""
    train and run our road segmentation algorithms and make a submission file
    this module calls functions from trainingMain and submissionMain
    All images of the training sets are used for training.
    choose a model and import the corresponding module ! 
    
"""

from trainingMain import train_classifier, doCV
from submissionMain import make_submission
import os

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

RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
DO_CV = False # run cross validation.


submission_filename = 'K_deep8_submission.csv'


modelNr = -1 #standard

def main():
    
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
    
    if DO_CV: 
        print("run cross validation")
        
        score = doCV(Model, 75, 25, 5)
        
        print("--------------------------------------")
        print('Resulting F1 Scores: ' + str(score)  )    
        print("--------------------------------------")
        
    print("start model")
    model = Model(modelNr)
    if RESTORE_MODEL:
        model.restoreModel()
    else:
        train_classifier(model, range(1, 101))
        
    make_submission(model, submission_filename)
    
    print('completed')
    
    
    
if __name__ == "__main__":
    main()
    
    
    