"""
@author: davidaponte

This script combines the predictions from SGD model 
and XGB model using weights from losses.
"""

# ensemble predictions by weighted average
import pandas as pd
import numpy as np

# load data
print("Loading Data")
submission = pd.read_csv("SubmissionFormat.csv", index_col=0)

xgb_model = pd.read_csv("predictions_XGB.csv")
  
​sgd_model = pd.read_csv("predictions_SGD.csv")

# losses from leaderboard
# L1 is sgdmodel, L2 is xgboost
print("Taking Weighted Average")
L_1, L_2 = 0.5139, 0.5976

w_1, w_2 = (1/L_1)/(1/L_1 + 1/L_2), (1/L_2)/(1/L_2 + 1/L_1)

# take weighted average
final_preds = (sgd_model * w_1) + (xgb_model * w_2)

# format predictions in dataframe
prediction_df = pd.DataFrame(columns=submission.columns,
                             index=submission.index,
                             data=final_preds)

# save prediction_df to csv
prediction_df.to_csv("SGD_XGB_predictions.csv")
print("Done!")
