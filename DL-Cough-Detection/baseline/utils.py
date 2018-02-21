#Authors: Filipe Barata, Maurice Weber, Kevin Kipfer

import glob
import random
import os, fnmatch, sys

import numpy as np
import pandas as pd
import librosa

ROOT_DIR = '../../Audio_Data'


def find_files(root, fntype, recursively=False):

	fntype = '*.'+fntype

	if not recursively:
		return glob.glob(os.path.join(root, fntype))

	matches = []
	for dirname, subdirnames, filenames in os.walk(root):
		for filename in fnmatch.filter(filenames, fntype):
			matches.append(os.path.join(dirname, filename))
	
	return matches


# def split_train_test_list():
# 	'''
# 	This procedure splits the data files stored in the root directory ROOT_DIR into test and training data
# 	'''


# 	# participants used in the test-set, generated at random
# 	listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] 
# 	list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
# 							'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']



# 	## Reading cough data
# 	print ('use data from root path %s'%ROOT_DIR)

# 	coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
# 	assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'


# 	# remove broken files
# 	for broken_file in list_of_broken_files:
# 		broken_file = os.path.join(ROOT_DIR, broken_file)
# 		if broken_file in coughAll:
# 			print ('file ignored: %s'%broken_file )
# 			coughAll.remove(broken_file)


# 	# split cough files into test- and training-set
# 	testListCough = []
# 	trainListCough = coughAll
# 	for name in coughAll:
# 		for nameToExclude in listOfParticipantsToExcludeInTrainset:
# 			if nameToExclude in name:
# 				testListCough.append(name)
# 				trainListCough.remove(name)

# 	print('nr of test samples coughing: %d' % len(testListCough))



# 	## Reading other data
# 	other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

# 	testListOther = []
# 	trainListOther = other
# 	for name in other:
# 		for nameToExclude in listOfParticipantsToExcludeInTrainset:
# 			if nameToExclude in name:
# 				testListOther.append(name)
# 				trainListOther.remove(name)

# 	print('nr of test samples NOT coughing: %d' % len(testListOther))

# 	return trainListCough, trainListOther, testListCough, testListOther



def extract_Signal_Of_Importance(file_name,
	window_size=0.08
	):
	'''
	Input:
		file_name the file to be loaded
		window_size the window to be cut out
	Output:
		A window of the audio signal of size window_size centered around the max absolute value of the entire sequence
	'''

	try:
		X, sample_rate = librosa.load(file_name)
	except Exception as e:
		print("An error occurred while parsing file ", file_name, " :\n")
		print(e)
		return [], -1

	maxValue = np.max(np.abs(X))
	absX = np.abs(X)
	indMax = absX.tolist().index(maxValue)
	numberOfSamples = np.ceil(sample_rate * window_size)
	startInd = int(np.max(indMax - numberOfSamples, 0))
	maxLeng = np.size(X)

	if startInd + 2*numberOfSamples > maxLeng - 1:
		endInd = int(maxLeng - 1)
		startInd = int(endInd - 2*numberOfSamples)
	else:
		endInd = int(startInd + 2*numberOfSamples)

	signal = X[startInd:endInd]
	return signal, sample_rate



if __name__ == "__main__":

	trainList, testList = split_train_test_list()

	print('train list length: %d'%len(trainList))
	print('test list length: %d'%len(testList))










































