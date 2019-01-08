# REFERENCES
# The idea of predicting iteratively by taking chunks of the full test set is from https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
# The weighting of class_99 in final predictions is discussed extensively on Kaggle but also present in this particular kernel amongst many others
#
# All of the differences between these files are simply to do with whether an array of clfs was passed or a single one, and whether feature engineering / scaling was done so that the test data could be put into a compatible format for making predictions

import time, gc; gc.enable()

import pandas as pd
import numpy as np; np.warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from feature_functions import agg_engineer_features_merge, agg_merge

directory = '/modules/cs342/Assignment2/'
# directory = '../Data/'

# Load full predictions and group down to one row of probabilities per object as required, then write to subm file
def write_final_predictions(filename, score, modeltype):
    final_predictions = pd.read_csv(filename)
    final_predictions = final_predictions.groupby('object_id').mean()
    print("Shape of predictions: {}".format(final_predictions.shape))
    final_predictions.to_csv('subm_' + modeltype + str(round(score,5)) +'.csv', index=True)



### NO FEATURES prediction functions

def predict_chunk_rfcbare(df, clf, metadata, aggs, features):
    
    # Merge chunk of test data with relevant metadata and aggregate
    full_test = agg_merge(df, metadata, aggs)
    full_test.fillna(0, inplace=True)

    # Make predictions
    predictions = clf.predict_proba(full_test[features])
            
    # Compute predictions_99 as the probability of class not being any of the others
    predictions_99 = np.ones(predictions.shape[0])
    for i in range(predictions.shape[1]):
        predictions_99 *= (1 - predictions[:, i])

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions, columns=['class_{}'.format(s) for s in clf.classes_])
    predictions_df['object_id'] = full_test['object_id']
    predictions_df['class_99'] = 0.14 * predictions_99 / np.mean(predictions_99)
    return predictions_df

def generate_predictions_rfcbare(clf, aggs, score, modeltype, features, chunks):
    
    start = time.time()
    metadata = pd.read_csv(directory + 'test_set_metadata.csv')
    filename = 'temp_' + modeltype + str(round(score,5)) + '.csv'

    remaining_data = None
    for chunk_num, df in enumerate(pd.read_csv(directory + 'test_set.csv', chunksize=chunks, iterator=True)):

        unique_ids = np.unique(df['object_id'])
        
        # Keep track of tail of the data to be predicted on after all of the chunks are executed
        new_remaining_data = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remaining_data is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remaining_data, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remaining_data = new_remaining_data
        
        predictions_df = predict_chunk_rfcbare(df, clf, metadata, aggs, features)
    
        if chunk_num == 0:
            predictions_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            predictions_df.to_csv(filename, header=False, mode='a', index=False)
    
        del predictions_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes'.format(chunks * (chunk_num + 1), (time.time() - start) / 60))        

    # Compute last object in remaining_data
    predictions_df = predict_chunk_rfcbare(remaining_data, clf, metadata, aggs, features)
    predictions_df.to_csv(filename, header=False, mode='a', index=False)
    write_final_predictions(filename, score, modeltype)



def generate_predictions_mlpbare(clfs, aggs, train_mean, ss, score, modeltype, features, chunks):

    start = time.time()
    metadata = pd.read_csv(directory + 'test_set_metadata.csv')
    filename = 'temp_' + modeltype + str(round(score,5)) + '.csv'

    for chunk_num, df in enumerate(pd.read_csv(directory + 'test_set.csv', chunksize=chunks, iterator=True)):
        
        # Merge chunk of test data with relevant metadata and aggregate, then fill NAs and scale
        full_test = agg_merge(df, metadata, aggs)
        full_test[features] = full_test[features].fillna(train_mean)
        full_test_ss = ss.transform(full_test[features])

        # Make predictions
        predictions = None
        for clf in clfs:
            if predictions is None:
                predictions = clf.predict_proba(full_test_ss)
            else:
                predictions += clf.predict_proba(full_test_ss)
        
        predictions = predictions / len(clfs)

        # Set predictions for the unseen class 99
        predictions_99 = np.ones(predictions.shape[0])
        for i in range(predictions.shape[1]):
            predictions_99 *= (1 - predictions[:, i])
        
        # Store predictions
        predictions_df = pd.DataFrame(predictions, columns=['class_{}'.format(s) for s in clfs[0].classes_])
        predictions_df['object_id'] = full_test['object_id']
        predictions_df['class_99'] = 0.14 * predictions_99 / np.mean(predictions_99)
        
        if chunk_num == 0:
            predictions_df.to_csv(filename,  header=True, mode='a', index=False)
        else: 
            predictions_df.to_csv(filename,  header=False, mode='a', index=False)
            
        del full_test, predictions_df, predictions
        if (chunk_num + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (chunk_num + 1), (time.time() - start) / 60))

    write_final_predictions(filename, score, modeltype)



### FEATURES prediction functions

def predict_chunk_rfcfeatures(df, clf, aggs, feature_spec, features, metadata):
    
    # Merge chunk of test data with relevant metadata and aggregate, then fill NAs
    full_test = agg_engineer_features_merge(df, metadata, aggs, feature_spec)
    full_test.fillna(0, inplace=True)

    # Make predictions
    predictions = clf.predict_proba(full_test[features])
            
    # Compute predictions_99 as the probability of a class not being any of the others
    predictions_99 = np.ones(predictions.shape[0])
    for i in range(predictions.shape[1]):
        predictions_99 *= (1 - predictions[:, i])

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions, columns=['class_{}'.format(s) for s in clf.classes_])
    predictions_df['object_id'] = full_test['object_id']
    predictions_df['class_99'] = 0.14 * predictions_99 / np.mean(predictions_99)
    return predictions_df

def generate_predictions_rfcfeatures(clf, aggs, feature_spec, score, modeltype, features, chunks):
    
    start = time.time()
    filename = 'temp_' + modeltype + str(round(score,5)) + '.csv'
    metadata = pd.read_csv(directory + 'test_set_metadata.csv')

    remaining_data = None
    for chunk_num, df in enumerate(pd.read_csv(directory + 'test_set.csv', chunksize=chunks, iterator=True)):

        unique_ids = np.unique(df['object_id'])
        
        # Keep track of tail of the data to be predicted on after all of the chunks are executed
        new_remaining_data = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remaining_data is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remaining_data, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remaining_data = new_remaining_data
        
        predictions_df = predict_chunk_rfcfeatures(df, clf, aggs, feature_spec, features, metadata)
        
        # File only requires a header to be written on the first iteration, otherwise append
        if chunk_num == 0:
            predictions_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            predictions_df.to_csv(filename, header=False, mode='a', index=False)
        del predictions_df
        gc.collect()

        print('{:15d} done in {:5.1f} minutes'.format(chunks * (chunk_num + 1), (time.time() - start) / 60))        

    # Compute predictions for the trailing data that didnt fit in a full chunk
    predictions_df = predict_chunk_rfcfeatures(remaining_data, clf, aggs, feature_spec, features, metadata)
    predictions_df.to_csv(filename, header=False, mode='a', index=False)
    write_final_predictions(filename, score, modeltype)



def predict_chunk_lgbm(df, clfs, aggs, feature_spec, features, metadata):
    
    # Merge chunk of test data with relevant metadata and aggregate, then fill NAs
    full_test = agg_engineer_features_merge(df, metadata, aggs, feature_spec)
    full_test.fillna(0, inplace=True)
    
    # Make predictions
    predictions = None
    for clf in clfs:
        if predictions is None:
            predictions = clf.predict_proba(full_test[features])
        else:
            predictions += clf.predict_proba(full_test[features])
            
    predictions = predictions / len(clfs)

    # Compute predictions_99 as the probability of class not being any of the others
    predictions_99 = np.ones(predictions.shape[0])
    for i in range(predictions.shape[1]):
        predictions_99 *= (1 - predictions[:, i])

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions, columns=['class_{}'.format(s) for s in clf.classes_])
    predictions_df['object_id'] = full_test['object_id']
    predictions_df['class_99'] = 0.14 * predictions_99 / np.mean(predictions_99)
    return predictions_df

def generate_predictions_lgbm(clfs, aggs, feature_spec, score, modeltype, features, chunks):
    
    start = time.time()
    filename = 'temp_' + modeltype + str(round(score,5)) + '.csv'
    metadata = pd.read_csv(directory + 'test_set_metadata.csv')

    remaining_data = None
    for chunk_num, df in enumerate(pd.read_csv(directory + 'test_set.csv', chunksize=chunks, iterator=True)):

        unique_ids = np.unique(df['object_id'])
        
        # Keep track of tail of the data to be predicted on after all of the chunks are executed
        new_remaining_data = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remaining_data is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remaining_data, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        
        # Create remaining samples df
        remaining_data = new_remaining_data
        
        predictions_df = predict_chunk_lgbm(df, clfs, aggs, feature_spec, features, metadata)
        
        # File only requires a header to be written on the first iteration, otherwise append
        if chunk_num == 0:
            predictions_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            predictions_df.to_csv(filename, header=False, mode='a', index=False)
        del predictions_df
        gc.collect()

        print('{:15d} done in {:5.1f} minutes'.format(chunks * (chunk_num + 1), (time.time() - start) / 60))        

    # Compute predictions for the trailing data that didnt fit in a full chunk
    predictions_df = predict_chunk_lgbm(remaining_data, clfs, aggs, feature_spec, features, metadata)
    predictions_df.to_csv(filename, header=False, mode='a', index=False)
    write_final_predictions(filename, score, modeltype)



def generate_predictions_mlpfeatures(clfs, aggs, feature_spec, train_mean, ss, score, modeltype, features, chunks):

    start = time.time()
    metadata = pd.read_csv(directory + 'test_set_metadata.csv')
    filename = 'temp_' + modeltype + str(round(score,5)) + '.csv'

    for chunk_num, df in enumerate(pd.read_csv(directory + 'test_set.csv', chunksize=chunks, iterator=True)):

        # Merge chunk of test data with relevant metadata and aggregate, then fill NAs and scale
        full_test = agg_engineer_features_merge(df, metadata, aggs, feature_spec)
        full_test[features] = full_test[features].fillna(train_mean)
        full_test_ss = ss.transform(full_test[features])

        # Make predictions
        predictions = None
        for clf in clfs:
            if predictions is None:
                predictions = clf.predict_proba(full_test_ss)
            else:
                predictions += clf.predict_proba(full_test_ss)
        
        predictions = predictions / len(clfs)
        print predictions

        # Set predictions for the unseen class 99
        predictions_99 = np.ones(predictions.shape[0])
        for i in range(predictions.shape[1]):
            predictions_99 *= (1 - predictions[:, i])
        
        # Store predictions
        predictions_df = pd.DataFrame(predictions, columns=['class_{}'.format(s) for s in clfs[0].classes_])
        predictions_df['object_id'] = full_test['object_id']
        predictions_df['class_99'] = 0.14 * predictions_99 / np.mean(predictions_99)
        
        if chunk_num == 0:
            predictions_df.to_csv(filename,  header=True, mode='a', index=False)
        else: 
            predictions_df.to_csv(filename,  header=False, mode='a', index=False)
            
        del full_test, predictions_df, predictions
        if (chunk_num + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (chunk_num + 1), (time.time() - start) / 60))

    write_final_predictions(filename, score, modeltype)
