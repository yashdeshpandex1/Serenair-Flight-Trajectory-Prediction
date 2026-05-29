# Imports
import joblib
from pathlib import Path

# get working scripts from preprocessing/
from data_cleaning import clean_df
from feature_engineering import feature_engineering
from sklearn_utilities import group_shuffle_split, scale_dataset 
from utilities import sort_values, num_aircrafts, initialise_features_and_target, initialise_df
from rnn_sequences import save_sequences

def prep_data(path='../data/opensky_raw.csv'):
    save_dir = Path('../data/rnn_data/') # define where feature scaler will be saved
    save_dir.mkdir(parents=True, exist_ok=True) # Create the folder if missing
    df = initialise_df(path) # read csv file
    df_sorted = sort_values(df) # sort the dataframe
    df_clean = clean_df(df_sorted) # clean the dataframe
    df_engineered = feature_engineering(df_clean) # feature engineering
    features, target = initialise_features_and_target() # features and target

    training_set, testing_set = group_shuffle_split(df=df_engineered) # shuffle split
    df_train, df_test, feature_scaler = scale_dataset(training_set, testing_set, features) # dataset scaling

    save_sequences(df_train, df_test, features, target)
    
    scaler_path = save_dir/'feature_scaler.joblib'
    joblib.dump(feature_scaler, scaler_path)

if __name__ == "__main__":
    prep_data()


