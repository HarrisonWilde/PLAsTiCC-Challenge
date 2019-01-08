# REFERENCES
# The feature extraction process is taken from this kernel https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss
# Justified in the report

import pandas as pd
import numpy as np; np.warnings.filterwarnings('ignore')

from tsfresh.feature_extraction import extract_features

# Group by object and calculate the passed aggregates
def agg_merge(df, df_meta, aggs):
    aggregate = df.groupby('object_id').agg(aggs)
    aggregate.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    aggregate = aggregate.reset_index()
    return aggregate.merge(right=df_meta, how='left', on='object_id')

# Extracting features from training data set, fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
def agg_engineer_features_merge(df, df_meta, aggs, feature_spec):

    df = process_flux(df)
    aggregate = df.groupby('object_id').agg(aggs)
    aggregate.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    

    # Extract some features (mentioned in report), mainly moments, ranges, means, and extrema of flux, flux ratios and passband data
    aggregate = process_flux_agg(aggregate)
    aggregate_ts_flux_passband = extract_features(df, 
        column_id='object_id', 
        column_sort='mjd', 
        column_kind='passband', 
        column_value='flux', 
        default_fc_parameters=feature_spec['flux_passband'])

    aggregate_ts_flux = extract_features(df, 
        column_id='object_id', 
        column_value='flux', 
        default_fc_parameters=feature_spec['flux'])

    aggregate_ts_flux_by_flux_ratio_sq = extract_features(df, 
        column_id='object_id', 
        column_value='flux_by_flux_ratio_sq', 
        default_fc_parameters=feature_spec['flux_by_flux_ratio_sq'])


    # This feature was suggested here and testified as being very useful https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    df_det = df[df['detected']==1].copy()
    aggregate_mjd = extract_features(df_det, 
        column_id='object_id', 
        column_value='mjd', 
        default_fc_parameters=feature_spec['mjd'])

    aggregate_mjd['mjd_diff_det'] = aggregate_mjd['mjd__maximum'].values - aggregate_mjd['mjd__minimum'].values
    del aggregate_mjd['mjd__maximum'], aggregate_mjd['mjd__minimum']
    

    # Bring all of the generated features together in one data frame before merging it with metadata and returning
    aggregate_ts_flux_passband.index.rename('object_id', inplace=True) 
    aggregate_ts_flux.index.rename('object_id', inplace=True) 
    aggregate_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True) 
    aggregate_mjd.index.rename('object_id', inplace=True)      
    aggregate_ts = pd.concat([aggregate, 
        aggregate_ts_flux_passband, 
        aggregate_ts_flux, 
        aggregate_ts_flux_by_flux_ratio_sq, 
        aggregate_mjd], axis=1).reset_index()
    return aggregate.merge(right=df_meta, how='left', on='object_id')

# Functions again present in https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss
# Simply generate some statistics surrounding flux to error ratios per passband as well as different forms of difference / mean
def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq, 
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq,}, 
        index=df.index)
    
    return pd.concat([df, df_flux], axis=1)

def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values
    
    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,       
        'flux_diff3': flux_diff / flux_w_mean,
        }, index=df.index)
    
    return pd.concat([df, df_flux_agg], axis=1)