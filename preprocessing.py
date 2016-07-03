# -*- coding: utf-8 -*-
"""
Preprocessing
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import constants


def label_columns(frame):
    frame.columns = constants.COL_NAMES


def stem_sentence(sentence):
    return " ".join(sorted([stemmer.stem(word) for word in sentence.lower().strip().split(" ")]))


def find_common(frame, column, n=15):
    """Print out a list of common answers"""
    return frame[column].groupby(frame[column].map(lambda x: str.lower(str(x)))).count().sort_values(ascending=False)[:n]


def vectorize_df(trainFrame, holdoutFrame):
    """Create dummy variables for categorical features and vectorize
    the data frames.  Returns the training 2D array, holdout 2D array,
    and list of feature names."""  
    trainDict = trainFrame.to_dict(orient='records')
    holdoutDict = holdoutFrame.to_dict(orient='records')    
    
    vec = DictVectorizer()
    vec.fit(trainDict)
    
    trainArray = vec.transform(trainDict).toarray()
    holdoutArray = vec.transform(holdoutDict).toarray()

    return trainArray, holdoutArray, vec.get_feature_names()


def dt_conversion_train(date_string):
    """Convert training data string into datetime. Anything less 
    than 30 is assumed to be 20XX."""
    b, y = date_string.split('-')
    y = int(y)
    if y < 30:
        Y = 2000 + y
    else:
        Y = 1900 + y
    dt = pd.to_datetime(b + str(Y), format="%b%Y")
    return dt


def dt_conversion_holdout(date_string):
    """Convert holdout data string into datetime."""
    x1, x2 = date_string.split('-')
    try:
        y = int(x1)
        Y = y + 2000
        b = x2
    except:
        y = int(x2)
        if y == 0:
            Y = 2000
        else:
            Y = y + 1900
        b = x1

    dt = pd.to_datetime(b + str(Y), format="%b%Y")
    return dt


def format_number_columns(frame):
    """Convert money and percentages to floats."""
    frame['revolving utilization'] = frame['revolving utilization'].map(lambda x: str(x).replace('%', '')).astype(float)
    frame['interest rate'] = frame['interest rate'].map(lambda x: str(x).replace('%', '')).astype(float)    
    frame['loan request'] = frame['loan request'].map(lambda x: str(x).replace('$', '').replace(',', '')).astype(float)    
    frame['loan funded'] = frame['loan funded'].map(lambda x: str(x).replace('$', '').replace(',', '')).astype(float)    
    frame['investor portion'] = frame['investor portion'].map(lambda x: str(x).replace('$', '').replace(',', '')).astype(float)


def convert_experience_to_num(frame):    
    """Convert years to float, with 10+ == 10, and <1 = 0."""
    frame['years employed'] = frame['years employed'].map(constants.EXPERIENCE_TO_NUM)


def convert_to_datetime(frame, holdout=False):    
    """Convert date strings to datetime.  These are formatted 
    differently in train and holdout sets."""
    if holdout:
        frame['issue date'] = pd.to_datetime(frame['issue date'], format="%y-%b")
        frame['earliest credit date'] = frame['earliest credit date'].map(dt_conversion_holdout)
    else:
        frame['issue date'] = frame['issue date'].map(dt_conversion_train)
        frame['earliest credit date'] = frame['earliest credit date'].map(dt_conversion_train)


def convert_grade_to_num(frame):
    """ Convert grade to numeric with A=1, B=2, ... and
    with subgrade:  A1=1.0, A2=1.2, B3=2.4, ..."""
    frame['loan grade'] = frame['loan grade'].map(constants.LOANGRADE_TO_NUM).astype(float)

    old = sorted(frame['loan subgrade'].value_counts().index)
    new = np.arange(1, 7.8, .2)
    subgradeMap = dict(zip(old, new))
    frame['loan subgrade'] = frame['loan subgrade'].map(subgradeMap).astype(float)


def stem_text(frame):    
    """Stem text in text fields."""
    for col in ['employer', 'loan title']:
        frame.ix[frame[col].isnull(), col] = 'nan'
        frame[col] = frame[col].map(stem_sentence)


def simplify_zip_code(frame):
    """Use the first digit for zip code, and treat as category (str)."""
    frame['zip code'] = frame['zip code'].astype(str).map(lambda x: x[0])


def simplify_employer(frame):
    """Aggregate occupations into a few categories."""
    #find_common(frame, 'employer')
    frame.ix[frame['employer'].isin(constants.VALID_EMPLOYER) == False, 'employer'] = 'other'


def simplify_loan_title(frame):
    """Aggregate titles into a few categories."""
    frame.ix[frame['loan title'].isin(constants.VALID_TITLES) == False, 'loan title'] = 'other'


def create_na_features(frame):
    """Before imputing values, it may be useful to know which records
    are missing values for particularly interesting columns, such as
    months since delinquency."""
    frame['has delinquency'] = 1
    frame.ix[frame['months since delinquency'].isnull(), 'has delinquency'] = 0
    frame['has prev record'] = 1
    frame.ix[frame['months since last record'].isnull(), 'has prev record'] = 0
    

def impute_values(frame):
    """Fill NA values with those determined from earlier analysis of
    the training set.  
    median:  [loan grade, loan subgrade, years employed, income, revolving util]
    9999:    [months since delinquency, months since last record] 
    BLANK:   [home status] """ 
    for col, fill in constants.FILL_VALUES.iteritems():
        frame[col] = frame[col].fillna(fill)
    

def create_features(frame):
    """Create new features from old ones."""
    frame['issue month'] = frame['issue date'].dt.month
    frame['issue month'] = frame['issue month'].astype(str)
    frame['issue year'] = frame['issue date'].dt.year
    frame['credit tenure'] = (frame['issue date'] - frame['earliest credit date']).astype('timedelta64[D]')
    frame['fraction funded'] = frame['loan funded'] / frame['loan request']
    frame['request over income'] = frame['loan request'] / frame['income']
    frame['funded over income'] = frame['loan funded'] / frame['income']
    frame['debt over income'] = frame['revolving balance'] / frame['income']
    frame['debt over request'] = frame['revolving balance'] / frame['loan request']
    frame['debt over funded'] = frame['revolving balance'] / frame['loan funded']
    frame['open line fraction'] = frame['open credit lines'] / frame['credit lines']


def preprocess(frame, holdout=False):
    # Remove bad rows
    if not holdout:
        # some rows (e.g. 364111) are missing a lot of data     
        frame = frame[frame.isnull().sum(axis=1) < 10]        
        # remove if missing the label:  interest rate == NA
        frame = frame[frame['interest rate'].notnull()]
    # Format the money and percent columns
    format_number_columns(frame)
    # Map years of experience categories to numeric scale
    convert_experience_to_num(frame)
    # Create datetimes
    convert_to_datetime(frame, holdout=holdout)
    # Map loan grades to numeric scale
    convert_grade_to_num(frame)
    # Stem text in text fields and sort alphabetically
    stem_text(frame)
    # Simplify the zip code to 1-digit str
    simplify_zip_code(frame)
    # Simplify the employer to 15 categories.
    simplify_employer(frame)
    # Simplify loan title to 9 categories
    simplify_loan_title(frame)
    # Create features indicating where NA is true
    create_na_features(frame)
    # Impute missing values
    impute_values(frame)
    # Create new features
    create_features(frame)
    # Drop columns we won't be using
    frame.drop(['borrower id', 'issue date', 'earliest credit date', 'reason for loan'], 
               axis=1, inplace=True)
   
    return frame


def main(): 
    """Read in training and holdout data, clean the data sets, convert to 
    2D arrays of numbers, and write to CSV."""
    # Read in data
    trainFrame = pd.read_csv(constants.TRAIN_PATH, encoding='utf-8')
    label_columns(trainFrame)
    holdoutFrame = pd.read_csv(constants.HOLDOUT_PATH, encoding='utf-8')
    label_columns(holdoutFrame)
    
    # Clean data
    trainFrame = preprocess(trainFrame)
    holdoutFrame = preprocess(holdoutFrame, holdout=True)

    # Vectorize the features with dummy variables
    trainArray, holdoutArray, colNames = vectorize_df(trainFrame, holdoutFrame) 
    print colNames
   
    # Convert back to dataframes
    cleanTrainFrame = pd.DataFrame(data=trainArray, columns=colNames)
    cleanHoldoutFrame = pd.DataFrame(data=holdoutArray, columns=colNames)

    # Write preprocessed data to CSV
    cleanTrainFrame.to_csv(constants.CLEAN_TRAIN_PATH, encoding='utf-8')        
    cleanHoldoutFrame.to_csv(constants.CLEAN_HOLDOUT_PATH, encoding='utf-8')


if __name__ == '__main__':
    main()