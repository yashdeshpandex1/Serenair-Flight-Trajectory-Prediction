# Imports
import joblib
from pathlib import Path

# get working scripts from preprocessing/
from data_cleaning import clean_df
from feature_engineering import feature_engineering
from sklearn_utilities import group_shuffle_split, scale_dataset 
from utilities import sort_values, num_aircrafts, initialise_features_and_target, initialise_df
from rnn_sequences import save_sequences, save_test_sequences
import argparse

def prep_train_data(path='../data/opensky_raw.csv'):
    save_dir = Path('../data/rnn_data/') # define where feature scaler will be saved
    save_dir.mkdir(parents=True, exist_ok=True) # Create the folder if missing
    df = initialise_df(path) # read csv file
    df_sorted = sort_values(df) # sort the dataframe
    df_clean = clean_df(df_sorted) # clean the dataframe
    df_engineered = feature_engineering(df_clean) # feature engineering
    features, target = initialise_features_and_target() # features and target

    training_set, testing_set = group_shuffle_split(df=df_engineered) # shuffle split
    df_train, df_test, feature_scaler, target_scaler = scale_dataset(training_set, testing_set, features, target) # dataset scaling

    save_sequences(df_train, df_test, features, target)
    
    scaler_path = save_dir/'feature_scaler.joblib'
    target_path = save_dir/'target_scaler.joblib'
    
    joblib.dump(feature_scaler, scaler_path)
    joblib.dump(target_scaler, target_path)

def prep_test_data(path='../data/opensky_test.csv'):
    save_dir = Path('../data/rnn_data/')
    save_dir.mkdir(parents=True, exist_ok=True)
    df_test = initialise_df(path)
    df_test = sort_values(df_test)
    df_test = clean_df(df_test)
    df_test = feature_engineering(df_test)
    
    features, target = initialise_features_and_target()
    df_test['raw_latitude'] = df_test['latitude']
    df_test['raw_longitude'] = df_test['longitude']
    
    scaler_path = save_dir / 'feature_scaler.joblib'
    target_path = save_dir / 'target_scaler.joblib'
    
    feature_scaler = joblib.load(scaler_path)
    target_scaler = joblib.load(target_path)
    
    df_test[features] = feature_scaler.transform(df_test[features])
    df_test[target] = target_scaler.transform(df_test[target])
    
    save_test_sequences(df_test, features, target, window_size=10)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run data pipelines')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help="Which pipeline to run: 'train' or 'test'?")
    parser.add_argument('--file', type=str, default=None,
                        help='Path to the CSV file (optional)')
    args = parser.parse_args()

    if args.mode == 'train':
        filepath = args.file if args.file else '../data/opensky_raw.csv'
        prep_train_data(path=filepath)
    elif args.mode == 'test':
        filepath = args.file if args.file else '../data/opensky_test.csv'
        prep_test_data(path=filepath)