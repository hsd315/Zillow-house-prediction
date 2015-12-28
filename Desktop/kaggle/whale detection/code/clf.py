# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:16:58 2015

@author: weizhi
"""

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn import grid_search
clf = SVC(probability=True,tol=0.01,gamma=0.1)

target = trainLabel['whaleID']



clf.fit(im_features, np.array(target))


#%% test feature 
#test_features
result = clf.predict_proba(test_features)

submit = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')
submit.iloc[:,1:] = result
submit.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/second_bagOfWords.csv',index = False)

training_names = '/Users/weizhi/Desktop/kaggle/whale detection/'
joblib.dump((clf, training_names, stdSlr, k, voc), training_names  +"/"+ "bof.pkl", compress=3)    
#%% training XGBoost

#X_train, X_valid,y_train,y_valid = train_test_split(X, yy, test_size=0.12, random_state=10)
import xgboost as xgb
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
yy = le.fit_transform(target)

dtrain = xgb.DMatrix(im_features, np.array(yy))
#dvalid = xgb.DMatrix(X_valid, y_valid)

#watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# train a XGBoost tree
print("Train a XGBoost model")
params = {"objective": "multi:softprob",
          "num_class":np.unique(yy).shape[0],
          "eta": 0.3,
          "max_depth": 10,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.95,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=400
num_boost_round =1000,
gbm = xgb.train(params, dtrain, num_trees)
#%% get the result
result = gbm.predict(xgb.DMatrix(test_features))

submit = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')
submit.iloc[:,1:] = result
submit.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/second_bagOfWords.csv',index = False)

training_names = '/Users/weizhi/Desktop/kaggle/whale detection/'
joblib.dump((gbm, training_names, stdSlr, k, voc), training_names  +"/"+ "bof.pkl", compress=3)  







