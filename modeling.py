# -*- coding: utf-8 -*-
"""
sklearn models
"""
from time import time

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.externals import joblib
    
import constants
import plotting


def standardize_features(X):
    from sklearn import preprocessing
    return preprocessing.scale(X)


def minmax_features(X):
    from sklearn import preprocessing
    mms = preprocessing.MinMaxScaler()
    return mms.fit_transform(X)


def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))


def benchmark(y):
    """Use the mean value of the training target as predictions
    for the validation set and calculate the RMSE."""
    y_train, y_validate = train_test_split(y, test_size=0.25, random_state=42)
    pred = np.zeros(len(y_validate)) + np.mean(y_train)
    return rmse(y_validate, pred)


def build_linear_regression(X, y):
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = linear_model.LinearRegression(normalize=False)
    t0 = time()
    clf.fit(X_train, y_train)
    print 'completed in: ', (time() - t0)
    #print clf.coef_
    y_pred = clf.predict(X_validate)
    return rmse(y_validate, y_pred), clf


def build_random_forest(X, y, min_samples_leaf=20, ntree=50, max_features=1.0):
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=42)    
    rf = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=min_samples_leaf, 
                               max_features=max_features, oob_score=True, n_jobs=-1)
    t0 = time()
    rf.fit(X_train, y_train)
    print 'completed in:', (time() - t0)
    y_pred = rf.predict(X_validate)
    return rmse(y_validate, y_pred), rf
    

def build_boosting_tree(X, y, depth=4, nsteps=20, eta=0.1):
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=42)
    gb = GradientBoostingRegressor(learning_rate=eta, n_estimators=nsteps, subsample=0.5, max_depth=depth)
    t0 = time()
    gb.fit(X_train, y_train)
    print 'completed in:', (time() - t0)
    y_pred = gb.predict(X_validate)
    return rmse(y_validate, y_pred), gb


def main():
    # Read in preprocessed data, and drop dubplicate index column
    cleanFrame = pd.read_csv(constants.CLEAN_TRAIN_PATH, encoding='utf-8')
    cleanFrame.drop(['Unnamed: 0', 'loan id'], axis=1, inplace=True)

    # Drop redundant features and and min-max normalize
    drop_cats = ['interest rate', 'state=WY', 'zip code=9', 'loan title=other', 
                 'loan category=other', 'home status=BLANK', 'payment length= 36 months', 
                 'initial status=w', 'employer=other', 'issue month=12', 'income status=not verified']
    X = cleanFrame.drop(drop_cats, axis=1).values
    #X = minmax_features(X)  # can't make predictions on holdout without minmax fit
    y = cleanFrame['interest rate'].values
    feature_names = cleanFrame.drop(drop_cats, axis=1).columns


    # Benchmarks
    #############

    # Build 0-feature benchmark model, using mean(y_train)
    benchmark_rmse = benchmark(cleanFrame['interest rate'].values)
    print benchmark_rmse
    
    # Build very simple linear model with a couple features
    limited_features = ['loan subgrade', 'issue year', 'payment length= 60 months',
                        'revolving utilization', 'recent inquiries']
    X_lim = cleanFrame[limited_features].values
    X_lim = minmax_features(X_lim)
    y_lim = cleanFrame['interest rate'].values
    simple_linear_rmse, clf = build_linear_regression(X_lim, y_lim)
    print simple_linear_rmse


    # Linear Model
    ###############
    
    # Full linear regression with all features
    full_linear_rmse, clf = build_linear_regression(X, y)
    print full_linear_rmse
    
    # Print coefficients
    feature_tup = zip(feature_names, clf.coef_)
    feature_tup.sort(key=lambda tup: abs(tup[1]))
    for tup in feature_tup:
        print tup

    plotting.plot_coef(clf, feature_names, ndisplay=20)

    # Write model to disk as pickle
    joblib.dump(clf, "linear_model/linear_model.pickle")
  

    # Random Forest Regression
    ############################

    rf_rmse, rf = build_random_forest(X, y, min_samples_leaf=6, ntree=300, max_features=0.3)
    print rf_rmse
  
    plotting.plot_feature_importance(rf, feature_names)

    # Write model to disk as pickle
    joblib.dump(rf, "random_forest/random_forest_model.pickle")


if __name__ == "__main__":
    main()
