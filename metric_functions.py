import pandas as pd
import numpy as np; np.warnings.filterwarnings('ignore')

# Credit to Olivier for providing an implementation of multi log-loss for the challenge
# code is slightly altered from his https://www.kaggle.com/ogrellier
def multi_weighted_logloss(y_true, y_predictions):

    # classes match https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194 these are clear when data is probed
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

    y_predictions = y_predictions.reshape(y_true.shape[0], len(classes), order='F')
    y_dummies = pd.get_dummies(y_true)

    # Normalize rows and limit y_predictions to 1e-15, 1-1e-15
    y_predictions = np.clip(a=y_predictions, a_min=1e-15, a_max=1 - 1e-15)
    y_predictions_log = np.log(y_predictions)

    # Get the log for ones, .values is used to drop the index of DataFrames
    y_log_ones = np.sum(y_dummies.values * y_predictions_log, axis=0)
    
    # Get the number of positives for each class
    num_positives = y_dummies.sum(axis=0).values.astype(float)

    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_weighted = y_log_ones * class_arr / num_positives

    loss = -np.sum(y_weighted) / np.sum(class_arr)

    return 'weighted log loss', loss, False