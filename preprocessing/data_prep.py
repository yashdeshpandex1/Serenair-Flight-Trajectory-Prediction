# Imports
import joblib
from pathlib import Path

# get working scripts from preprocessing/
from data_cleaning import clean_df
from feature_engineering import feature_engineering
from sklearn_utilities import group_shuffle_split, scale_dataset 
from utilities import sort_values, num_aircrafts, \
    initialise_features_and_target, initialise_df, integrate_weather_data
from rnn_sequences import save_sequences
from fetch_weather_data import get_weather_data
import argparse
import pandas as pd

def prep_train_data(path='../data/opensky_raw.csv', task='next_instance'):
    """Complete data pipeline to from feature engineering to creating sequences,
    collecting and integrating weather data, shuffle split and saving scaler files.

    Args:
        path (str): File needed to process. Defaults to '../data/opensky_raw.csv'.
        task (str): Choosing prediction task. Defaults to 'next_instance'.
    """
    
    save_dir = Path(f'../data/rnn_data_{task}/') # define where feature scaler will be saved
    save_dir.mkdir(parents=True, exist_ok=True) # Create the folder if missing
    df = initialise_df(path) # read csv file
    df_sorted = sort_values(df) # sort the dataframe
    df_clean = clean_df(df_sorted) # clean the dataframe
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], unit='s')
    
    center_lat = df_clean['latitude'].mean()
    center_lon = df_clean['longitude'].mean()
    start_date = df_clean['timestamp'].min().strftime('%Y-%m-%d')
    end_date = (df_clean['timestamp'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    df_weather = get_weather_data(center_lat, center_lon,
                                start_date, end_date)
    df_clean = integrate_weather_data(df_clean, df_weather)
    
    df_engineered = feature_engineering(df_clean) # feature engineering
    features, target = initialise_features_and_target() # features and target

    training_set, testing_set = group_shuffle_split(df=df_engineered) # shuffle split
    df_train, df_test, feature_scaler, target_scaler = scale_dataset(training_set, testing_set, features, target) # dataset scaling

    save_sequences(df_train, df_test, features, target, task=task)
    
    scaler_path = save_dir/f'feature_scaler_{task}.joblib'
    target_path = save_dir/f'target_scaler_{task}.joblib'
    
    joblib.dump(feature_scaler, scaler_path)
    joblib.dump(target_scaler, target_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run data pipelines')
    parser.add_argument('--task', type=str, required=True, choices=['next_instance', 'next_ten_mins'],
                        help="Which pipeline to run: 'next instance?' or 'next ten mins?'?")
    parser.add_argument('--file', type=str, default=None,
                        help='Path to the CSV file (optional)')
    args = parser.parse_args()
    filepath = args.file if args.file else '../data/opensky_raw.csv'
    prep_train_data(path=filepath, task=args.task)