# -*- coding: utf-8 -*-
"""
Constants used for preprocessing and such.
"""
# Data paths
META_PATH = "C:\Users\Tom\Desktop\state_farm\Metadata.csv"
TRAIN_PATH = "C:\Users\Tom\Desktop\state_farm\Data for Cleaning & Modeling.csv"
CLEAN_TRAIN_PATH = "C:\Users\Tom\Desktop\state_farm\clean_train.csv"
HOLDOUT_PATH = "C:\Users\Tom\Desktop\state_farm\Holdout for Testing.csv"
CLEAN_HOLDOUT_PATH = "C:\Users\Tom\Desktop\state_farm\clean_holdout.csv"

CATEGORICAL_COLS = ["payment length", "employer", "home status", "income status", 
                    "loan category", "loan title", "zip code", "state", 
                    "initial status", "issue month",]

VALID_TITLES = [u'consolid debt', u'card credit refinanc', u'home improv', u'consolid',
               u'consolid debt loan', u'card consolid credit',
               u'major purchas', u'loan person', u'consolid loan', u'busi',
               u'card credit payoff', u'card credit', u'card credit refin',
               u'expens medic', u'person', u'card credit off pay', u'car financ',
               u'loan', u'payoff', u'vacat', u'debt', u'freedom', u'loan my', ]

VALID_EMPLOYER = ['teacher', 'manag', 'nurs regist', 'supervisor', 'sale', 'rn',
                  'driver', 'owner', 'manag project', 'manag offic', 'manag gener',
                  'driver truck', ]

FILL_VALUES = {'loan grade': 3.0,
               'loan subgrade': 3.0,
               'years employed': 6.0, 
               'income': 63000.0,
               'revolving utilization': 57.9,
               'months since delinquency': 31.0,
               'months since last record': 80.0,
               'home status': 'BLANK', }

EXPERIENCE_TO_NUM = {u'< 1 year': 0,
                     u'1 year': 1,
                     u'10+ years': 10,
                     u'2 years': 2,
                     u'3 years': 3,
                     u'4 years': 4,
                     u'5 years': 5,
                     u'6 years': 6,
                     u'7 years': 7,
                     u'8 years': 8,
                     u'9 years': 9,}

LOANGRADE_TO_NUM = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}

COL_NAMES = ['interest rate',
             'loan id',
             'borrower id',
             'loan request',
             'loan funded',
             'investor portion',
             'payment length',
             'loan grade',
             'loan subgrade',
             'employer',
             'years employed',
             'home status',
             'income',
             'income status',
             'issue date',
             'reason for loan',
             'loan category',
             'loan title',
             'zip code',
             'state',
             'payment to income ratio',
             'late payments',
             'earliest credit date',
             'recent inquiries',
             'months since delinquency',
             'months since last record',
             'open credit lines',
             'derogatory public records',
             'revolving balance',
             'revolving utilization',
             'credit lines',
             'initial status']