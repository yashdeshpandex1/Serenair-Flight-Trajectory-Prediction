# Imports
import sys
import os
import pandas as pd
import numpy as np
import joblib

# get working scripts from ../preprocessing/
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from preprocessing.data_cleaning import clean_df
from preprocessing.feature_engineering import feature_engineering
from preprocessing.sklearn_utilities import group_shuffle_split, scale_dataset 
from preprocessing.utilities import sort_values, num_aircrafts, initialise_features_and_target, initialise_df
from rnn_sequences import save_sequences

def prep_data(path='../data/opensky_raw.csv'):
    df = initialise_df(path) # read csv file
    df_sorted = sort_values(df) # sort the dataframe
    df_clean = clean_df(df_sorted) # clean the dataframe
    df_engineered = feature_engineering(df_clean) # feature engineering
    features, target = initialise_features_and_target() # features and target

    training_set, testing_set = group_shuffle_split(df=df_engineered) # shuffle split
    df_train, df_test, feature_scaler = scale_dataset(training_set, testing_set, features) # dataset scaling

    save_sequences(df_train, df_test, features, target)
    joblib.dump(feature_scaler, 'feature_scaler.joblib')

if __name__ == "__main__":
    prep_data()


