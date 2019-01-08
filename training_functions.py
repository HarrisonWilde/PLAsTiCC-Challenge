# REFERENCES
# Use of sample weights in RF and LGBM classification came from here https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
# This refers to the following section of the code in both functions:
#   w = y.value_counts()
#   weights = {i : np.sum(w) / w[i] for i in w.index}

import itertools, os, time; os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np; np.warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from metric_functions import multi_weighted_logloss


# Random forest cross validated training
def training_RF_cv(params, full_train, y, folds, modeltype):

    # Initialise timer, param result array
    start_time = time.time()
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
    param_results = []

    keys, values = zip(*params.items())
    for param in [dict(zip(keys, v)) for v in itertools.product(*values)]:

        param_time = time.time()
        print '\n' + str(param)
        all_fold_predictions = np.zeros((len(full_train), np.unique(y).shape[0]))

        # Utilise the passed Stratified K-Fold to cross validate during training, with each parameter combo in the grid search
        for fold, (train, val) in enumerate(folds.split(y, y)):

            train_x, train_y = full_train.iloc[train], y.iloc[train]
            val_x, val_y = full_train.iloc[val], y.iloc[val]

            # Define the classifier with passed parameters, set n_jobs to -1 for parallel processing and then fit it on the current folds split of training and validation data
            clf = RandomForestClassifier(**param)
            clf.set_params(n_jobs = -1)
            clf.fit(train_x, train_y, sample_weight=train_y.map(weights))

            # Record predictions and score for each fold
            all_fold_predictions[val, :] = clf.predict_proba(val_x)

            fold_result = param.copy()
            fold_result['fold'] = fold + 1
            fold_result['score'] = multi_weighted_logloss(val_y, all_fold_predictions[val, :])[1]

            param_results.append(fold_result)
            print 'Fold ' + str(fold + 1) + ', time elapsed for this parameter so far:', round(time.time() - param_time, 1)

        param_result = param.copy()
        param_result['fold'] = 'avg'
        param_result['score'] = multi_weighted_logloss(y, all_fold_predictions)[1]

        param_results.append(param_result)

        print 'Score for this parameter combination:', param_result['score']
        print 'Total time (in minutes) elapsed so far:', round((time.time() - start_time) / 60, 1)

    results = pd.DataFrame(param_results)

    # The minimum recorded score is found alongside its associated parameters, then these parameters are used to fit a single classifier on all of the training data to return
    min_score_row = results.sort_values('score', ascending=True).loc[results['fold'] == 'avg'].iloc[0]
    min_score_params = min_score_row[params.keys()].to_dict()
    clf = RandomForestClassifier(**min_score_params)
    clf.set_params(n_jobs = -1, verbose = 1)
    clf.fit(full_train, y, sample_weight=y.map(weights))
    score = min_score_row['score']

    # Records the results of the grid search for plotting purposes and output finalised clf parameters
    results.to_csv('results_' + modeltype + str(round(score,5)) + '.csv', index = False)
    print(clf)
    return clf, score


# Multi-Layer Perceptron cross validated training
def training_MLP_cv(full_train_ss, y, folds, modeltype):

    # Initialise timer, array of clfs, loss curves and an empty np array of zeroes to accept prediction probabilities
    start_time = time.time()
    clfs = []
    all_fold_predictions = np.zeros((len(full_train_ss), np.unique(y).shape[0]))
    loss_curves = []

    # Utilise the passed Stratified K-Fold to cross validate during training
    for fold, (train, val) in enumerate(folds.split(y,y)):

        train_x, train_y = full_train_ss.iloc[train], y.iloc[train]
        val_x, val_y = full_train_ss.iloc[val], y.iloc[val]

        # Define the model along with its characteristics, namely hidden layer sizes, alpha, iteration number and tolerance with which to end early
        # clf = MLPClassifier(hidden_layer_sizes=(512,256,128,64,32,16), max_iter=600, alpha=0.00005, solver='sgd', tol=1e-9, verbose=10000)
        clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32), max_iter=600, alpha=1e-4, solver='sgd', tol=1e-8, verbose=10000)
        clf.fit(train_x, train_y)

        # Keep track of loss curves during training, prediction probabilities and the score for this fold
        loss_curves.append(clf.loss_curve_)
        all_fold_predictions[val, :] = clf.predict_proba(val_x)
        print(multi_weighted_logloss(val_y, all_fold_predictions[val, :])[1])
        
        clfs.append(clf)
        print 'Fold ' + str(fold + 1) + ', time elapsed so far:', round(time.time() - start_time, 1)

    # Overall score, and recording of loss curves to a file
    score = multi_weighted_logloss(y, all_fold_predictions)[1]
    pd.DataFrame(loss_curves).to_csv(modeltype + str(score) + 'loss_curves.csv')
    print('Overall score: ' + str(round(score,5)))
    return clfs, score


# LGBM cross validated training
def training_LGBM_cv(params, full_train, y, folds):

    # Initialise array of clfs, weights and an empty np array of zeroes to accept prediction probabilities
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
    all_fold_predictions = np.zeros((len(full_train), np.unique(y).shape[0]))
    clfs = []

    # Utilise the passed Stratified K-Fold to cross validate during training
    for fold, (train, val) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[train], y.iloc[train]
        val_x, val_y = full_train.iloc[val], y.iloc[val]
    
        # Define the classifier using passed parameters and then fit it using the custom multi-weighted log-loss function as the evaluation metric
        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric=multi_weighted_logloss,
            verbose=1000, early_stopping_rounds=50, sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        # Record the predictions for this fold and output the associated score
        all_fold_predictions[val, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold + 1, multi_weighted_logloss(val_y, all_fold_predictions[val, :])[1]))
    
    # Overall score
    score = multi_weighted_logloss(y, all_fold_predictions)[1]
    print('Overall loss: ' + str(round(score,5)))
    return clfs, score