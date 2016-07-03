# -*- coding: utf-8 -*-
"""
Loading models / holdout, and making predictions.
"""
import pandas as pd
from sklearn.externals import joblib

import constants

def main():
    # Read in preprocessed data
    holdoutFrame = pd.read_csv(constants.CLEAN_HOLDOUT_PATH, encoding='utf-8')
    loanIDs = holdoutFrame['loan id']
    holdoutFrame.drop(['Unnamed: 0', 'loan id'], axis=1, inplace=True)

    drop_cats = ['interest rate', 'state=WY', 'zip code=9', 'loan title=other', 
                 'loan category=other', 'home status=BLANK', 'payment length= 36 months', 
                 'initial status=w', 'employer=other', 'issue month=12', 'income status=not verified']
    X = holdoutFrame.drop(drop_cats, axis=1).values
    
    # Read in models
    clf = joblib.load('linear_model/linear_model.pickle')
    rf = joblib.load('random_forest/random_forest_model.pickle')

    # Make predictions
    lm_pred = clf.predict(X)
    lm_pred = pd.Series(lm_pred)
    lm_pred.name = 'Pred Interest Rate LM'
    
    rf_pred = rf.predict(X)
    rf_pred = pd.Series(rf_pred)
    rf_pred.name = 'Pred Interest Rate RF'

    print lm_pred.mean(), rf_pred.mean()     
    
    # Write to CSV
    results = pd.concat([loanIDs, lm_pred, rf_pred], axis=1)
    results.to_csv('Results from Thomas Hettinger.csv')


if __name__ == "__main__":
    main()