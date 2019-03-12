"""
@author: davidaponte

This script contains the XGBoost model I used. Credit goes to 
https://marielgh.github.io/boxplots.html for inspiring the way I set
up the training loop. 
"""

######### xgboost ###############
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk import RegexpTokenizer
from scipy import sparse
import xgboost as xgb

# load training and test datasets and sample submission
print("Loading Data")
train_df = pd.read_csv('boxplot_train.csv', index_col=0)
test_df = pd.read_csv('boxplot_test.csv', index_col=0)
submission = pd.read_csv('SubmissionFormat.csv', index_col=0)

# label columns
LABELS = ['Function','Object_Type','Operating_Status','Position_Type','Pre_K', 'Reporting',
                'Sharing','Student_Type', 'Use']
# feature columns
NON_LABELS = ['FTE','Facility_or_Department', 'Function_Description','Fund_Description',
                       'Job_Title_Description', 'Location_Description','Object_Description',
                       'Position_Extra', 'Program_Description', 'SubFund_Description',
                       'Sub_Object_Description', 'Text_1', 'Text_2','Text_3','Text_4', 'Total']

# train and test features for holdout validation
train_ = train_df[NON_LABELS]
test_ = test_df[NON_LABELS]

# one hot encode labels for training set
y_train = pd.get_dummies(train_df[LABELS])
    

# drop FTE and Total columns
train_ = train_.drop(['FTE', 'Total'], axis = 1)
test_ = test_.drop(['FTE', 'Total'], axis = 1)


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
train_x_ = combine_text_columns(train_)
# test
test_x_ = combine_text_columns(test_)

# preprocess text
lem = WordNetLemmatizer()

def preprocess(s):
    """
    This function preprocesses the text for vectorization.
    """
    s = s.replace( "K-", "K" )
    tokenizer = RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    tokens = tokenizer.tokenize(s.lower())
    stem_words = [lem.lemmatize(word) for word in tokens]
    return " ".join(stem_words)

# apply  
x_train_processed = train_x_.apply(preprocess)
x_test_processed = test_x_.apply(preprocess)


# initialize TFIDF + Count Vec
tfidf = TfidfVectorizer(ngram_range=(1, 2), 
                        max_df=1.0, 
                        min_df=10,
                       stop_words='english')

cnt_vec = CountVectorizer(max_df=1.0,
                         min_df=10,
                         stop_words='english')

print("Feature Extraction")
# fit 
tfidf.fit(x_train_processed)
cnt_vec.fit(x_train_processed)

# transform 
X_tfidf = tfidf.transform(x_train_processed)
X_test_tfidf = tfidf.transform(x_test_processed)

X_cnt_vec = cnt_vec.transform(x_train_processed)
X_test_cnt_vec = cnt_vec.transform(x_test_processed)


####################### with dimensionality reduction ############################
# k=500
print("Dimensionality Reduction")
kbest = SelectKBest(chi2, k=500)

Xt_train = kbest.fit_transform(X_tfidf, y_train.values)
Xt_test = kbest.transform(X_test_tfidf)

Xc_train = kbest.fit_transform(X_cnt_vec, y_train.values)
Xc_test = kbest.transform(X_test_cnt_vec)

# hstack
X_train = sparse.hstack((Xt_train, Xc_train))
X_test = sparse.hstack((Xt_test, Xc_test))

################### XGBOOST ##################################
print("Modeling")
b=[0,37,48,51,76,79,82,87,96,104]

predictions = np.ones((X_test.shape[0], 104))

for i in np.arange(9):
    
    # print round to keep track
    print('Round: '+ str(i+1))
    # slice of labels
    w = b[i+1]-b[i]
    # params
    opt_params = {'booster' : 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'learning_rate': 0.2,
              'n_estimators':1000,
              'colsample_bytree': 0.3,
              'max_depth':5,
              'min_child_weight':32,
              'reg_lambda':1,
              'subsample':0.9,
              'num_class' : w}
    # slice y_train
    y_train_i = y_train.iloc[:,b[i]:b[i+1]].values
    # index with max prob
    y_train_i = np.argmax(y_train_i, axis=1)
    
    # dtrain, dtest                 
    dtrain = xgb.DMatrix(X_train, label=y_train_i)
    dtest = xgb.DMatrix(X_test)
                     
    # train 200 epochs             
    model_xgb = xgb.train(opt_params, dtrain, 200)
    # append predictions to pred_prob matrix                 
    predictions[:,b[i]:b[i+1]] = model_xgb.predict(dtest,ntree_limit=model_xgb.best_ntree_limit).reshape(X_test.shape[0],w)
    
# format predictions in dataframe
xgb_model = pd.DataFrame(columns=submission.columns,
                             index=submission.index,
                             data=predictions)
# save csv
xgb_model.to_csv("predictions_XGB.csv")
print("Done!")