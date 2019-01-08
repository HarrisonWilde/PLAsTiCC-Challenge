import pandas as pd
import numpy as np; np.warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from feature_functions import agg_merge
from training_functions import training_MLP_cv
from prediction_functions import generate_predictions_mlpbare

directory = '/modules/cs342/Assignment2/'
# directory = '../Data/'

modeltype = 'mlpbare'

aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean'],
}

train_meta = pd.read_csv(directory + 'training_set_metadata.csv')
train_data = pd.read_csv(directory + 'training_set.csv')

# merge the two datasets loaded in above and aggregate
train_merged = agg_merge(train_data, train_meta, aggs)

# Necessary code when scaling the data using StandardScaler(), creates new scaled data removing all unnecessary columns for predictions / training
y = train_merged['target']
train_merged.drop(['target', 'object_id', 'distmod', 'hostgal_specz', 'decl', 'gal_l', 'gal_b', 'ddf'], 1, inplace=True)
train_mean = train_merged.mean(axis=0)
train_merged.fillna(train_mean, inplace=True)
train_merged_temp = train_merged.copy()
standard_scaler = StandardScaler()
train_merged_scaled = pd.DataFrame(standard_scaler.fit_transform(train_merged_temp), columns=train_merged_temp.columns)

# Define number of folds and train the model
folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=1)
clfs, score = training_MLP_cv(train_merged_scaled, y, folds, modeltype)

generate_predictions_mlpbare(clfs, aggs, train_mean, standard_scaler, score, modeltype, features=train_merged_scaled.columns, chunks=5000000)
