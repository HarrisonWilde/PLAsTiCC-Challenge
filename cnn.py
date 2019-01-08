# REFERENCES
# https://www.kaggle.com/aantonova/cnn-signals-as-images/notebook
# The setup for this model is inspired by the efforts of the contributor above, namely in the BatchGenerator approach and decisions made around how to put the time series data into bins
# Note that due to me rushing at the end, I didn't manage to modularise this code and put it into the nice separate function files as I did with the others, apologies for that

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import StratifiedKFold
from datetime import datetime as dt

import gc
import time
import warnings
warnings.simplefilter(action = 'ignore')

# Try removing tensorflow import if it doesn't run, but I imagine it will work as I assume many people will be using keras and tensorflow
import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Flatten, Dropout, Dense
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

directory = '/modules/cs342/Assignment2/'
# directory = '../Data/'

modeltype = 'cnn'

# Make aggregations through grouping on passband
def get_statistics(df):
    groups = df.groupby('passband')
    result = groups['mjd'].apply(np.count_nonzero).values
    result = np.vstack((result, groups['mjd'].apply(np.asarray).apply(lambda x: np.median(x[1:] - x[:-1]))))
    return np.transpose(result)

# Data augmentation technique, create inputs for CNN
def timeseries_to_bins(ndar, step):
    warnings.simplefilter(action = 'ignore')

    # First recorded time for an object
    start = np.min(ndar[:, 0])

    # Total observed duration for object
    mjd_length = np.max(ndar[:, 0]) - start

    # Count of bins for an object's time series, this try block was required as sometimes the step seemed to be NaN for reasons unknown, this was a rare occurrence and so should not have a massive effect on predictions
    try:
        timeseries_length = int(float(mjd_length) / float(step)) + 1
    except:
        timeseries_length = int(float(mjd_length) / float(5.0)) + 1
    
    # Matrix to store counts in each bin for each row
    cnt = np.zeros((6, timeseries_length))

    # Matrix for the result with 3 channels: flux, flux_err, detected
    result = np.zeros((6, timeseries_length, 3))
    
    # Calculating sums of rows in the source array of data
    for i in range(ndar.shape[0]):
        row = ndar[i, :]
        try:
            col_num = int((row[0] - start) / step)
        except:
            col_num = int((row[0] - start) / 5.0)
        cnt[int(row[1]), col_num] += 1
        result[int(row[1]), col_num, 0] += row[2]
        result[int(row[1]), col_num, 1] += row[3]
        result[int(row[1]), col_num, 2] += row[4]
        
    # Calculate mean values
    result[:, :, 0] /= cnt
    result[:, :, 1] /= cnt
    result[:, :, 2] /= cnt
    
    # Normalise the flux channels by row
    for channel in range(2):
        means = np.reshape([np.mean(result[i, ~np.isnan(result[i, :, channel]), channel]) for i in range(6)] * timeseries_length, (6, timeseries_length), order='F')
        stds = np.reshape([np.std(result[i, ~np.isnan(result[i, :, channel]), channel]) for i in range(6)] * timeseries_length, (6, timeseries_length), order='F')
        result[:, :, channel] = (result[:, :, channel] - means) / stds
        
    # Replace any NaNs that have slipped through with zeroes
    result = np.nan_to_num(result)
    return result

# Create BatchGenerator objects accepting data, target (if applicable) a max_length dimension and the size of batch to generate using these parameters
class BatchGenerator(Sequence):
    
    def __init__(self, X, y, max_length, batch_size = 32, predict = False):
        self.X = X
        self.index = list(X['object_id'].unique())
        self.y = y
        self.max_length = max_length
        self.batch_size = batch_size
        self.predict = predict

        if not predict:
            self.on_epoch_end()
        
    def __getitem__(self, index_batch):
        idx = self.index[index_batch * self.batch_size : (index_batch + 1) * self.batch_size]
        batch = np.zeros((len(idx), 6, self.max_length, 3))
        if not self.predict:
            target = np.zeros((len(idx), self.y.shape[1]))
        
        for i, obj in enumerate(idx):
            ndar = self.X[self.X['object_id'] == obj][['mjd', 'passband', 'flux', 'flux_err', 'detected']]
            stats = get_statistics(ndar) # for defining step size
            data = timeseries_to_bins(ndar.values, np.median(stats[:, 1]))
            
            if data.shape[1] < self.max_length:
                data = np.concatenate((data, np.zeros((6, self.max_length - data.shape[1], 3))), axis = 1)
                
            batch[i] = data
            if not self.predict:
                target[i] = self.y.loc[obj].values

        if self.predict:
            return batch
        else:
            return batch, target
        
    def on_epoch_end(self):
        if not self.predict:
            np.random.shuffle(self.index)
        
    def __len__(self):
        if self.predict:
            return int(np.ceil(len(self.index) / self.batch_size))
        else:
            return int(len(self.index) / self.batch_size)

# Model definition, CNN with 3 layers and 3 dense fully connected layers after flattening
def get_model(class_num, input_shape, dropout = .5):
    # Define the model
    model = Sequential()

    # Add convolutional layers separated by batch normalisation and leaky relu activational layers
    model.add(Conv2D(filters = 8, kernel_size = 1, padding = 'same', 
        use_bias = False, 
        kernel_initializer = RandomNormal(), 
        kernel_regularizer = l2(0.01), 
        input_shape=input_shape))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(LeakyReLU(alpha = 0.3))
    model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', 
        use_bias = False, 
        kernel_initializer = RandomNormal(), 
        kernel_regularizer = l2(0.01)))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', 
        use_bias = False, 
        kernel_initializer = RandomNormal(),
        kernel_regularizer = l2(0.01)))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(LeakyReLU(alpha = 0.1))

    # Flattening section
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())

    # Fully connected dense section of the model, 3 dense layers of decreasing size
    model.add(Dense(class_num * 4, 
        kernel_initializer = RandomNormal(),
        kernel_regularizer = l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0))
    model.add(Dropout(dropout))
    model.add(Dense(class_num * 2, 
        kernel_initializer = RandomNormal(), 
        kernel_regularizer = l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0))
    model.add(Dropout(dropout))
    # softmax for output ensure multi class prob distribution is returned, values sum to 1
    model.add(Dense(class_num, 
        kernel_initializer = RandomNormal(), 
        activation = 'softmax'))
    print(model.summary())
    return model

# Cross-validation for model
def cv_scores(num_folds, classes, X_train, y_train, max_length, early_stopping = -1, n_epoch = 100, batch_size = 32):
    def lr_schedule_cosine(x):
        return .001 * (np.cos(np.pi * x / n_epoch) + 1.) / 2
    warnings.simplefilter('ignore')
    print("Starting cross-validation at {} with random_state {}".format(time.ctime(), 0))
    folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 0)
    # Create arrays to store results
    train_pred = pd.DataFrame(columns = classes, index = y_train['object_id'])
    valid_pred = pd.DataFrame(columns = classes, index = y_train['object_id'])
    y = pd.get_dummies(y_train.set_index('object_id')['target']).reset_index()
    histories = {}
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(np.zeros(y_train.shape[0]), y_train.set_index('object_id'))):
        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))
        # Preparing data for training by taking subsets according to folds, getting list of unique objects in these folds, and creating BatchGenerators for training and validation
        train_y = y.iloc[train_idx]
        train_objects = train_y['object_id'].values
        train_x = X_train[X_train['object_id'].isin(train_objects)]
        valid_y = y.iloc[valid_idx]
        valid_objects = valid_y['object_id'].values
        valid_x = X_train[X_train['object_id'].isin(valid_objects)]
        train_gen = BatchGenerator(train_x, train_y.set_index('object_id'), max_length, batch_size = batch_size)
        valid_gen = BatchGenerator(valid_x, valid_y.set_index('object_id'), max_length, batch_size = batch_size)

        # Get the model
        model = get_model(len(classes), (6, max_length, 3))
        # Categorical crossentropy aligns with the log loss function defined in the competition for this purpose
        model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy' , metrics = ['categorical_accuracy'])
        model_file =  'fold_' + str(n_fold) + '.h5'
        # Set a function to change the learning rate throughout training, also generate model checkpoints to be able to recover process if training is interrupted and to also maintain the weights of the model trained across multiple folds
        callbacks = [
            LearningRateScheduler(lr_schedule_cosine),
            ModelCheckpoint(filepath = model_file, monitor = 'val_loss', save_best_only = True, save_weights_only = True)
        ]
        if early_stopping > 0:
            callbacks.append(EarlyStopping(monitor = 'val_loss', patience = early_stopping))
        
        # Fitting model
        model.fit_generator(train_gen, validation_data = valid_gen, callbacks = callbacks, epochs = n_epoch)
        histories[n_fold] = model.history.history

        # Prediction for train and valid data
        train_gen = BatchGenerator(train_x, None, max_length, batch_size = 1, predict = True)
        valid_gen = BatchGenerator(valid_x, None, max_length, batch_size = 1, predict = True)
        model.load_weights(model_file)
        train_pred.loc[train_objects] = pd.DataFrame(model.predict_generator(train_gen), columns = classes, index = train_objects) 
        valid_pred.loc[valid_objects] = pd.DataFrame(model.predict_generator(valid_gen), columns = classes, index = valid_objects)

    print(histories)
    return model, train_pred, valid_pred, histories

# Altered version of the one present in metric_functions.py to work with the different data formats present in this approach
def weighted_multiclass_logloss(y_true, y_pred):
    class_weights = [1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
    y_pred_clip = np.clip(a = y_pred, a_min = 1e-15, a_max = 1 - 1e-15)
    loss = np.sum(y_true * y_pred_clip.applymap(np.log), axis = 0)
    loss /= np.sum(y_true, axis = 0)
    loss *= class_weights
    return -(np.sum(loss) / np.sum(class_weights))


# Load in the data and set target and classes explicitly
train_data = pd.read_csv(directory + 'training_set.csv')
train_meta = pd.read_csv(directory + 'training_set_metadata.csv')
target = train_meta[['object_id', 'target']]
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

# Calculate maximum image length using mjd
max_length = -1
for obj in train_meta['object_id']:
    ndar = train_data[train_data['object_id'] == obj][['mjd', 'passband', 'flux', 'flux_err', 'detected']]
    stats = get_statistics(ndar)
    data = timeseries_to_bins(ndar.values, np.median(stats[:, 1]))
    if data.shape[1] > max_length:
        max_length = data.shape[1]

print('Count of columns in image:', max_length)

# Train the model
model, train_pred, valid_pred, histories = cv_scores(num_folds = 2, 
    n_epoch = 100, 
    classes = classes, 
    X_train = train_data, 
    y_train = target,
    max_length = max_length)

y = pd.get_dummies(target.set_index('object_id')['target'])

score = weighted_multiclass_logloss(y, valid_pred)

print('Score for train:', weighted_multiclass_logloss(y, train_pred))
print('Score for valid:', score)


# PREDICTIONS, code is an adaptation of the format found in prediction_functions.py

def predict_chunk(df, model):
    chunk_objects = np.unique(df['object_id'].unique())
    df.fillna(0, inplace=True)
    df_generator = BatchGenerator(df, None, max_length, batch_size = 1, predict = True)
    predictions = pd.DataFrame(model.predict_generator(df_generator), columns = ['class_{}'.format(s) for s in classes], index = chunk_objects) 
    predictions_99 = np.ones(predictions.shape[0])
    for i in range(predictions.shape[1]):
        predictions_99 *= (1 - predictions.iloc[:,i]) 
    predictions['object_id'] = chunk_objects
    predictions['class_99'] = 0.14 * predictions_99 / np.mean(predictions_99)
    return predictions

start = time.time()
test_meta = pd.read_csv(directory + 'test_set_metadata.csv')
filename = 'temp_' + modeltype + str(round(score,5)) + '.csv'

chunks = 5000000

# Go through test data in chunks and call predict_chunk on each
remaining_data = None
for chunk_num, df in enumerate(pd.read_csv(directory + 'test_set.csv', chunksize=chunks, iterator=True)):
    unique_ids = np.unique(df['object_id'])
    new_remaining_data = df.loc[df['object_id'] == unique_ids[-1]].copy()
    if remaining_data is None:
        df = df.loc[df['object_id'].isin(unique_ids[:-1])]
    else:
        df = pd.concat([remaining_data, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
    remaining_data = new_remaining_data
    predictions_df = predict_chunk(df, model)
    if chunk_num == 0:
        predictions_df.to_csv(filename, header=True, mode='a', index=False)
    else:
        predictions_df.to_csv(filename, header=False, mode='a', index=False)
    del predictions_df
    gc.collect()
    print('{:15d} done in {:5.1f} minutes'.format(chunks * (chunk_num + 1), (time.time() - start) / 60))        

# Compute last objects in remaining_data
predictions_df = predict_chunk(remaining_data, model)
predictions_df.to_csv(filename, header=False, mode='a', index=False)
predictions_df = pd.read_csv(filename)

# Reshape to ensure one prediction for each object and write to file
final_predictions = predictions_df.groupby('object_id').mean()
print("Shape of predictions: {}".format(final_predictions.shape))
final_predictions.to_csv('subm_' + modeltype + str(round(score,5)) + '.csv', index=True)