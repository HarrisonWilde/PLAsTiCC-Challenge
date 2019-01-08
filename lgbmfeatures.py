import sys, os, argparse, time; os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gc; gc.enable()

import pandas as pd
import numpy as np
np.warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

from feature_functions import agg_engineer_features_merge
from training_functions import training_LGBM_cv
from prediction_functions import generate_predictions_lgbm

directory = '/modules/cs342/Assignment2/'
# directory = '../Data/'

modeltype = 'lgbm'

aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum', 'skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
}

# Feature dictionary from https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss
feature_spec = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    },
    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
    },
    'flux_passband': {
        'fft_coefficient': [
                {'coeff': 0, 'attr': 'abs'}, 
                {'coeff': 1, 'attr': 'abs'}
            ],
        'kurtosis' : None, 
        'skewness' : None,
    },
    'mjd': {
        'maximum': None, 
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}

# Features inspired by https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'learning_rate': 0.03,
    'subsample': .9,
    'colsample_bytree': 0.5,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'n_estimators': 1000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3
}

train_meta = pd.read_csv(directory + 'training_set_metadata.csv')
train_data = pd.read_csv(directory + 'training_set.csv')

# Create features and aggregate statistics, as well as merging the two datasets loaded in above
train_merged = agg_engineer_features_merge(train_data, train_meta, aggs, feature_spec)

# Define the training target and prepare the data for training by removing NAs
y = train_merged['target']
train_merged.drop('target', 1, inplace=True)
train_merged.fillna(0, inplace=True)

# Define number of folds and train the model
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
clfs, score = training_LGBM_cv(params, train_merged, y, folds)

# Use the trained models to generate predictions for the test set, to be submitted
generate_predictions_lgbm(clfs, aggs, feature_spec, score, modeltype, features=train_merged.columns, chunks=5000000)