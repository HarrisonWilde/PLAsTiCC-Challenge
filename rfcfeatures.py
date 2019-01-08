import pandas as pd
import numpy as np; np.warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

from feature_functions import agg_engineer_features_merge
from training_functions import training_RF_cv
from prediction_functions import generate_predictions_rfcfeatures

directory = '/modules/cs342/Assignment2/'
# directory = '../Data/'

modeltype = 'rfcfeatures'

aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum','skew'],
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

# Different params to grid search over
param_grid = { 
    'n_estimators' : [80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300],
    'max_features' : ['auto', 'log2'],
    'max_depth' : [8,9,10,11,12],
    'criterion' :['entropy']
}

train_meta = pd.read_csv(directory + 'training_set_metadata.csv')
train_data = pd.read_csv(directory + 'training_set.csv')

# Create features and aggregate statistics, as well as merging the two datasets loaded in above
train_merged = agg_engineer_features_merge(train_data, train_meta, aggs, feature_spec)

y = train_merged['target']
train_merged.drop('target', 1, inplace=True)
train_merged.fillna(0, inplace=True)

# Define number of folds and train_data the model
folds = StratifiedKFold(n_splits=30, shuffle=True, random_state=1)
clf, score = training_RF_cv(param_grid, train_merged, y, folds, modeltype)

# Use the trained models to generate predictions for the test set, to be submitted
generate_predictions_rfcfeatures(clf, aggs, feature_spec, score, modeltype, features=train_merged.columns, chunks=5000000)