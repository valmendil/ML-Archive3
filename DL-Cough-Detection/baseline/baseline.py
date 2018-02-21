#Authors: Main Writer Maurice Weber, (Help Kevin Kipfer)

import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from utils import *


# hyperparameters
pca_components = 10
n_trees = 500
max_depth = 10
max_features = 12
n_freq = 1025
time_frames = 7



def stack_spectrograms(filenames,
	print_every_n_steps=50
	):
	'''
	Input:
		A list of filenames
	Output:
		A matrix X with vectorized spectrograms as rows
		A list labels with entries 1 (cough sound) and 0 (other sound)
	'''

	count = 0

	for fn in filenames:

		# get signal of importance
		Signal, sr = extract_Signal_Of_Importance(fn)
		if np.size(Signal) / sr != 0.16:
			continue
		
		stft = np.abs(librosa.stft(Signal))

		# reshape spectrogram into row vector
		row_vec = np.empty((0))
		for i in range(np.shape(stft)[1]):
			row_vec = np.hstack([row_vec, stft[:,i]])

		# normalize vector
		row_vec /= np.linalg.norm(row_vec)

		# get label
		label = int("Coughing" in fn)
		
		if count == 0:
			X = row_vec
			labels = [label]
		else:
			X = np.vstack([X, row_vec])
			labels.append(label)
		
		count += 1

		if count % print_every_n_steps == 0:
			print(count, " files processed.")

	return labels, X



def generate_cough_model(trainListCough,
	trainListOther,
	testListCough,
	testListOther,
	batch_size = 256):
	'''
	Input:
		Lists with filenames for training and test data, cough, and non-cough sounds
	Output:
		list of labels for training and test data
		feature matrices cough_model_train and cough_model_test
	'''
	
	assert batch_size % 2 == 0

	batch_size_half = batch_size // 2

	max_idx =  min(len(trainListCough), len(trainListOther)) - batch_size_half - 1
	counter = 0

	print("computing cough model for training data...")

	for idx in range(0, max_idx, batch_size_half):

		counter += 1
		end = idx + batch_size_half
		train_batch_cough = trainListCough[idx:end]
		train_batch_other = trainListOther[idx:end]
		train_batch_cough.extend(train_batch_other)

		train_filenames = train_batch_cough

		print('processing batch %d'%counter)
		labels_train, X_train = stack_spectrograms(train_filenames)

		## compute model
		# PCA on vectorized spectrograms
		X_train = sklearn.preprocessing.scale(X_train, axis=0, with_mean=True, with_std=False)
		pca = decomposition.PCA()
		pca.n_components = pca_components
		X_reduced_train = pca.fit_transform(X_train)

		# Decompress data with inverse PCA transform
		X_projected_train = pca.inverse_transform(X_reduced_train)
		residual_error_train = np.mean((X_train - X_projected_train) ** 2, axis = 1)

		# reconstruct spectrogram to compute energy features for training set
		for i in range(np.shape(X_projected_train)[0]):

			stft_reduced = np.reshape(X_projected_train[i,:], (n_freq, time_frames))
			
			# compute energy
			energy = np.mean(librosa.feature.rmse(S=stft_reduced))
			if i == 0:
				energy_features_train = energy
			else:
				energy_features_train = np.vstack([energy_features_train, energy])

		# merge features into single data matrix
		cough_model_train_ = np.column_stack((X_reduced_train, energy_features_train, residual_error_train))

		if counter == 1:
			cough_model_train = cough_model_train_
			all_train_labels = labels_train
		else:
			cough_model_train = np.vstack([cough_model_train, cough_model_train_])
			all_train_labels.extend(labels_train)



	## compute model for test data
	print("computing cough model for test data...")
	
	testListCough.extend(testListOther)
	test_filenames = testListCough

	labels_test, X_test = stack_spectrograms(test_filenames)

	# PCA on vectorized spectrograms
	X_test = sklearn.preprocessing.scale(X_test, axis=0, with_mean=True, with_std=False)
	pca = decomposition.PCA()
	pca.n_components = pca_components
	X_reduced_test = pca.fit_transform(X_test)

	# Decompress data with inverse PCA transform
	X_projected_test = pca.inverse_transform(X_reduced_test)
	residual_error_test = np.mean((X_test - X_projected_test) ** 2, axis = 1)

	# reconstruct spectrogram to compute energy features for test set
	for i in range(np.shape(X_projected_test)[0]):

		stft_reduced = np.reshape(X_projected_test[i,:], (n_freq, time_frames))
		
		# compute energy
		energy = np.mean(librosa.feature.rmse(S=stft_reduced))
		if i == 0:
			energy_features_test = energy
		else:
			energy_features_test = np.vstack([energy_features_test, energy])


	# merge features into single data matrix
	cough_model_test = np.column_stack((X_reduced_test, energy_features_test, residual_error_test))


	return all_train_labels, cough_model_train, labels_test, cough_model_test



def split_train_test_list():
	'''
	This procedure splits the data files stored in the root directory ROOT_DIR into test and training data
	'''


	# participants used in the test-set, generated at random
	listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] 
	list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
							'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']



	## Reading cough data
	print ('use data from root path %s'%ROOT_DIR)

	coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
	assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'


	# remove broken files
	for broken_file in list_of_broken_files:
		broken_file = os.path.join(ROOT_DIR, broken_file)
		if broken_file in coughAll:
			print ('file ignored: %s'%broken_file )
			coughAll.remove(broken_file)


	# split cough files into test- and training-set
	testListCough = []
	trainListCough = coughAll
	for name in coughAll:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListCough.append(name)
				trainListCough.remove(name)

	print('nr of test samples coughing: %d' % len(testListCough))



	## Reading other data
	other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

	testListOther = []
	trainListOther = other
	for name in other:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListOther.append(name)
				trainListOther.remove(name)

	print('nr of test samples NOT coughing: %d' % len(testListOther))

	return trainListCough, trainListOther, testListCough, testListOther




if __name__ == "__main__":

	# get lists for datafiles; split into training and test sets
	trainListCough, trainListOther, testListCough, testListOther = split_train_test_list()

	# compute cough model
	y_train, X_train, y_test, X_test = generate_cough_model(trainListCough, trainListOther, 
															testListCough, testListOther)

	# train random Forest Classifier
	rf = RandomForestClassifier(n_estimators = n_trees, max_features=max_features, max_depth=max_depth)
	rf.fit(X_train, y_train)

	predictions_test = rf.predict(X_test)
	predictions_train = rf.predict(X_train)

	train_accuracy = np.sum(predictions_train == y_train)/len(X_train)
	test_accuracy = np.sum(predictions_test == y_test)/len(X_test)

	aucroc_score_test = roc_auc_score(y_test, predictions_test)
	aucroc_score_train = roc_auc_score(y_train, predictions_train)

	# print accuracy
	print('*********  RESULTS *********')
	print('test accuracy: %f'%test_accuracy)
	print('train accuracy: %f'%train_accuracy)
	print('auc roc score test: %f'%aucroc_score_test)
	print('auc roc score train: %f'%aucroc_score_train)

























