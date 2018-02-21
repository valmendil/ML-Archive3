import csv
from datetime import datetime
import numpy as np
import math
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn import linear_model
from xgboost import XGBRegressor


import warnings

RUN_BASELINE   = True
RUN_MAIN       = True
DO_CV1         = False
DO_CV2         = True
RUN_SUBMISSION = True


PATH='./data'
TRAIN='data_train'
TEST='data_test'


with warnings.catch_warnings():
    	warnings.simplefilter("ignore")

	###############################################################################################
	#preprocessing
	###############################################################################################


	print ('preprocess the data ...')

	def extract(file_name):
		weather = {}
		X = [] #features
		Y = [] #label
		dates=[]

		#load weather
		with open('%s/weather.csv'%PATH, 'rb') as csvfile:
		     reader = csv.reader(csvfile, delimiter=',')
		     reader.next()
		     for row in reader:
			 date = row[0]
			 date = datetime.strptime(date, '%Y-%m-%d')
			 weather[date] = [float(s) if s != '' else 0. for s in row[1:] ]

		#load data + labels
		with open('%s/%s.csv'%(PATH,file_name), 'rb') as csvfile:
		     reader = csv.reader(csvfile, delimiter=',')
		     reader.next()
		     for row in reader:
			 data = []
			 label = row[-1]

		         #add dates
			 date = row[0]
			 dates.append(date)                 
			 date = datetime.strptime(date, '%Y-%m-%d')
			 weekday  = date.weekday() #probably more people on weekends?
			 day = date.timetuple().tm_yday #nth day of the year
			 week = int(date.strftime("%V")) #same week?
			 year = date.year - 2005 
			 month = date.month
			 data.extend([weekday, day, week, year, month])

		         #convert
		         row = [float(s) for s in row[1:-1]]

		         #add features0:10 + weather - as is
			 data.extend(row[1:9])
			 data.extend(row[10:13])
			 data.extend(weather[date])

		         #add holidays
			 def one_hot_holiday(r):
				if r==3:
				   return [1,1]
				elif r==2:
				   return [1,0]
				elif r==1:
				   return [0,1]
				else:
				   return [0,0]
			 b_holiday = row[0]
			 s_holiday = row[9]
			 data.extend(one_hot_holiday(b_holiday))
			 data.extend(one_hot_holiday(s_holiday))
			 
			 X.append(data)
			 Y.append(label)
			 
		X = np.array(X)
		
		#handle labels
		if Y[0] !='':
		   Y = [float(y) for y in Y]
		   Y = np.array(Y)
		else: #missing
		   Y = np.zeros_like(np.array(Y))
		


	  	return X, Y, dates

	#preprocess data
	x,y, _ = extract(TRAIN)
	scaler = RobustScaler()
	x = scaler.fit_transform(x)
	poly = PolynomialFeatures(2)
	x = poly.fit_transform(x)

	#split in train + test set
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	print ('- number of features:%s'%x.shape[1])
	print ('- number of examples in total:%s'%x.shape[0])
	print ('- number of examples used for training:%s'%x_train.shape[0])
	print ('- number of examples used for testing:%s'%x_test.shape[0])
	print 
	print ('-----------------------------------------------------------------------')
	print 


	###############################################################################################
	#train baseline
	###############################################################################################

	def rmse(targets, predictions):
		return np.sqrt(((predictions - targets) ** 2).mean())


	if RUN_BASELINE:
		baseline_model = linear_model.Lasso(tol=0.1) #tol extra large, such that the optimization can converge

		# Set the parameters by cross-validation
		tuned_parameters = [{'alpha':np.logspace(-3, 3, 100)}]


		print("# Tuning hyper-parameters for the baseline")
		print

		clf = GridSearchCV(baseline_model, tuned_parameters, cv=3, refit=True, verbose=1) 
		clf.fit(x_train, y_train)

		print
		print("Best parameters set found on development set:")
		print(clf.best_params_)
		print
		print("Grid scores on development set (r2):")
		print
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
			      % (mean, std * 2, params))
		print
		print("Result:")
		print

		score = rmse(y_test, clf.predict(x_test))
		print("RMSE: %.4f" % score)
		print
		print('-----------------------------------------------------------------------')
		print



	###############################################################################################
	#train main model
	###############################################################################################


	if RUN_MAIN:

		#use all features:

		clf = XGBRegressor(         seed= 0, 
		                            silent=1,
		                            colsample_bytree=0.8,
		                            learning_rate=0.1,
		                            max_depth=4,
		                            min_child_weight=1,
		                            subsample=0.8,
		                            n_estimators= 150,
		                            reg_alpha=0,  
		                            max_delta_step= 0,
		                            booster= 'dart' 
		                            )

		tuned_parameters = [{
		        'n_estimators': [100, 150, 200, 250],
		        'max_depth': [3, 4, 5],
		        'colsample_bytree': [0.8, 0.9, 1.0],
		        'subsample': [0.8,0.9, 1], 
		        'learning_rate': [0.1, 0.01, 0.2, 0.02], 
		        'reg_alpha':[0, 0.1,0.2], 
		        'min_child_weight':[1,2,3]
		    }]
		        

		print("# Tuning hyper-parameters for all features")
		print

		if DO_CV1:
			clf = GridSearchCV(clf, tuned_parameters, cv=3, n_jobs=20, refit=True, verbose=1) 

		clf.fit(x_train, y_train)


		if DO_CV1:
			print
			print("Grid scores on development set (r2):")
			print
			means = clf.cv_results_['mean_test_score']
			stds = clf.cv_results_['std_test_score']
			for mean, std, params in zip(means, stds, clf.cv_results_['params']):
				print("%0.3f (+/-%0.03f) for %r"
				      % (mean, std * 2, params))
			print
			print("Best parameters set found on development set:")
			print(clf.best_params_)

			# ...
			#Best parameters set found on development set:
			#{'reg_alpha': 0, 'colsample_bytree': 0.8, 'learning_rate': 0.1, 'min_child_weight': 1, 'n_estimators': 150, 'subsample': 0.8, 'max_depth': 4}
			#Detailed report:
			#RMSE: 241.1478

		else: print('skip cv ... (set DO_CV1 to True if you wish to use it)')


		print
		print("Result:")
		print


		score = rmse(y_test, clf.predict(x_test))
		print("RMSE: %.4f" % score)
		print
		print('-----------------------------------------------------------------------')
		print
	
		# feature selection: Fit model using each importance as a threshold
		thresholds = sorted(set([ round(elem, 3) for elem in clf.feature_importances_]), reverse=True)[:100]
		best_thresh, best_score, best_n = 0, 1000, 0
		for thresh in thresholds:
			# select features using threshold
			selection = SelectFromModel(clf, threshold=thresh, prefit=True)
			select_x_train = selection.transform(x_train)
			# train model
			selection_model =  XGBRegressor(    seed= 0, 
						            silent=1,
						            booster= 'dart'   
						        )
			selection_model.fit(select_x_train, y_train)
			# eval model
			select_x_test = selection.transform(x_test)
			y_pred = selection_model.predict(select_x_test)
			score = rmse(y_test, y_pred)
			print("Thresh=%.3f, n=%d, RMSE: %.2f%%" % (thresh, select_x_test.shape[1], score))
		        if best_score >= score:
				best_score = score           
				best_thresh=thresh
				best_n = select_x_test.shape[1]
		print ("-> best score for %d features and a threshold of %.3f"%(best_n, best_thresh))
		print
		print ('retrain xgboost with the selected features')
		selection = SelectFromModel(clf, threshold=best_thresh, prefit=True)
		x_train = selection.transform(x_train)

		#retrain with entire dataset
		model = XGBRegressor(       seed= 0, 
		                            silent=1
		                            )

		if DO_CV2:
			print ('Gridsearch on dev.set ...')
			tuned_parameters2 = {	
				'n_estimators': [50, 80, 100, 125, 150, 175, 200, 250, 500, 1000],
				'max_depth': [2,3, 4, 5, 6], 
				'subsample': [0.7, 0.8, 0.9, 1], 
				'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25], 
				'reg_alpha':[0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25], 
				'gamma': [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
				}

			#model = GridSearchCV(model, tuned_parameters2, cv=5, n_jobs=20, refit=True, verbose=0)
			
			n_iter_search = 10000
			model = RandomizedSearchCV(model, tuned_parameters2, n_iter=n_iter_search, n_jobs=20, refit=True, verbose=1, random_state=0) 

		model.fit(x_train, y_train)

		if DO_CV2:
			print("Best parameters set found on development set:")
			print(model.best_params_)
			model = model.best_estimator_ #want to refit the best estimator later on all the data

		print('Result:')
		x_test = selection.transform(x_test)
		score = rmse(y_test, model.predict(x_test))
		print("RMSE: %.4f" % score)
		print
		print('-----------------------------------------------------------------------')
		print



		###############################################################################################
		#Generate Submition 
		###############################################################################################

		if RUN_SUBMISSION:

			print ('retrain xgb-model with the entire training set')
			x = selection.transform(x)
			model.fit(x, y)

			print ('load validation set')
			x_val, _, dates = extract(TEST)
			x_val = scaler.transform(x_val)
			x_val = poly.transform(x_val)

			print('make prediction')
			x_val = selection.transform(x_val)
			predictions=model.predict(x_val)

			#write submission 
			with open('submission.csv', 'w') as csvfile:
			    fieldnames = ['date', 'prediction']
			    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			    writer.writeheader()
			    for i in range(len(dates)):
				writer.writerow({fieldnames[0]: dates[i], fieldnames[1]: predictions[i]})

print ('done')

	
