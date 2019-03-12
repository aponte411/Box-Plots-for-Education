
"""
Created on Wed Feb  6 08:07:30 2019

@author: davidaponte

This script was inspired by Cooper Stainbrook, isms, and pjbull
from DataCamp and DataDriven.org

isms: https://github.com/drivendata/boxplots-for-education-1st-place
Cooper Stainbrook: https://github.com/drivendataorg/box-plots-for-education/tree/master/2nd-place/code
pjbull: https://github.com/drivendataorg/box-plots-sklearn/tree/master/src

"""
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


# load training and test datasets and submission
print("Loading data")
train_df = pd.read_csv('Boxplots_train.csv', index_col=0)
test_df = pd.read_csv('Boxplots_test.csv', index_col=0)
submission = pd.read_csv('SubmissionFormat.csv', index_col=0)

# list of labels
LABELS = ['Function','Object_Type','Operating_Status','Position_Type','Pre_K', 'Reporting',
                'Sharing','Student_Type', 'Use']
# list of features
NON_LABELS = ['FTE','Facility_or_Department', 'Function_Description','Fund_Description',
                       'Job_Title_Description', 'Location_Description','Object_Description',
                       'Position_Extra', 'Program_Description', 'SubFund_Description',
                       'Sub_Object_Description', 'Text_1', 'Text_2','Text_3','Text_4', 'Total']

# train and test features
train_ = train_df[NON_LABELS]
test_ = test_df[NON_LABELS]
# labels
labels = train_df[LABELS]

# encode labels
label_encoder = LabelEncoder()
for i in range(labels.shape[1]):
    labels.iloc[:,i] = label_encoder.fit_transform(labels.iloc[:,i])

# drop FTE and Total columns
train_ = train_.drop(['FTE', 'Total'], axis = 1)
test_ = test_.drop(['FTE', 'Total'], axis = 1)

#############################################
# Feature extraction
# Here I will combine TFIDF and Hashing vectorizers.
# I got the idea from Cooper Stainbrook and DataCamp.org
#############################################


# create combine text function
def combine_text_columns(dataframe, to_drop=LABELS):
    """ 
    Converts all text in each row into a single vector.
    
    params:
    dataframe = dataframe read in using pandas
    to_drop = all non-text columns and labels (optional)
    
    """
    
    # drop non-text columns and labels
    drop = set(to_drop) & set(dataframe.columns.tolist())
    text_data = dataframe.drop(drop, axis=1)
    
    # replace missing values with blank space
    text_data = text_data.fillna('')
    
    # join all text per row using a space and lowercase
    return text_data.apply(lambda x: " ".join(x).lower(), axis=1)


# create a single new column for cleaned text data
# train    
x_ = combine_text_columns(train_)
# test
test_x_ = combine_text_columns(test_)

#token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# initialize TFIDF vectorizer and Hashing Vectorizer
tfidf = TfidfVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                        ngram_range=(1, 2), 
                        max_df=1.0, 
                        min_df=10,
                        stop_words='english')

hsv = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                        stop_words='english')

# fit tfidf and hashing vectorizer to train data
print("Feature Extraction")
tfidf.fit(x_)
hsv.fit(x_)

# transform 
X_tfidf = tfidf.transform(x_)
X_test_tfidf = tfidf.transform(test_x_)

X_hsv = hsv.transform(x_)
X_test_hsv = hsv.transform(test_x_)

# combine
X = sparse.hstack((X_hsv, X_tfidf))
X_test = sparse.hstack((X_test_hsv, X_test_tfidf))

######################################################
# Using SGDClassifier to build 27 models. Each class 
# (9 total) will have 3 models - varying some of the 
# hyperparameters such as:
#    - alpha
#    - penalty
#####################################################
print("Modeling")
preds1 = []
preds2 = []
preds3 = []

for i in range(labels.shape[1]):
    # print label number to keep track
    print("label = ", i)
    # first classifier with low alpha
    sgd1 = SGDClassifier(loss = 'log', n_iter = 120, alpha = 0.000001)
    # second classifier with L1 regularization 
    sgd2 = SGDClassifier(loss = 'log', n_iter = 120, penalty = 'l1')
    # third classifier with L2 regularization
    sgd3 = SGDClassifier(loss = 'log', n_iter = 120, penalty = 'l2')
    # train models
    sgd1.fit(X, labels.iloc[:,i].astype(int))
    sgd2.fit(X, labels.iloc[:,i].astype(int))
    sgd3.fit(X, labels.iloc[:,i].astype(int))
    # for first iteration store predictions
    if i == 0:
        preds1 = sgd1.predict_proba(X_test)
        preds2 = sgd2.predict_proba(X_test)
        preds3 = sgd3.predict_proba(X_test)
    # after, hstack them    
    else:
        preds1 = np.hstack((preds1,sgd1.predict_proba(X_test)))
        preds2 = np.hstack((preds2,sgd2.predict_proba(X_test)))
        preds3 = np.hstack((preds3,sgd3.predict_proba(X_test)))

# average predictions
preds = (preds1 + preds2 + preds3)/3.0

# format predictions in dataframe
prediction_df = pd.DataFrame(columns=submission.columns,
                             index=submission.index,
                             data=preds)


# save prediction_df to csv
prediction_df.to_csv("predictions_SGD.csv")
print("Done!")

