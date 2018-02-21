CIL project 2016 Road Segmentation

TEAM: 
==========================================

	0xdeadbeef


AUTHORS:
=========================================

	Kipfer Kevin

	Erdin Matthias

	Dykcik Lukasz


Required Software: 
==========================================

	Python 3.4
	NumPy 1.11.0
	scikit-image 0.9.3
	scikit-learn 0.17.1
	Pillow 2.3.0
	Tensorflow 0.8.0 (only for NNModel)


How To Run:
==========================================

Path to Root folder:
------------------------------------------

	CIL/src/main

Path to images:
------------------------------------------

	the training data needs to be stored in the following way:
		the test data needs to be stored in the folder CIL/data/test_set_images/
			each image needs to be in a seperate folder (the same way as we received the test data)
		all the training images need to be in the same folder CIL/data/training/images/
		all the ground truth images need to be in the same folder CIL/data/training/groundtruth/
	
	
Run:
------------------------------------------
	In order to run training and prediction from the commandline, do the following:
		1. Enter directory src/main (`cd src/main`)
		2. Set PYTHONPATH accordingly (`export PYTHONPATH=../mask:../Models:../Utils`)
		3. Execute run.py (`python run.py`)

	The file allows to specify in line 33 what model shall be used. The following Model_id's are possible:
		1 -> NNModel                 (really simple cnn baseline)
        	2 -> exampleSKLModel         (simple baseline using mean and variance as features and logistic regression as estimator)
        	3 -> colorProbabilityModel   (tries to predict the color probability cp )
        	4 -> leafModel               (first layer of our approach)
        	5 -> deepModel               (contains the code of the second layer and runs both layers of our approach)

	In line 42 one can specify the filename of the submission. It will be stored in the folder CIL/submissions.

	The overlay images, the CP images and the Hough images of the final are produced while it's running and stored in the folder CIL/predictions
	
Analysis	
------------------------------------------
	In order to produce the analysis images that show the accuracy of the prediction, one has to run the model first since we need the prediction images 
	next type  make -j8 into a console at path CIL/	
	the images will be stored in the folder CIL/analysis
