# Utilities needed to transform our data gathered by getting_data.ipynb

# Imports
import numpy as np
import pandas as pd

def initialise_df(path: str) -> pd.DataFrame:
    """
    Simply initialises a dataframe from the path provided.
    """
    df = pd.read_csv(path)
    return df

# one plane (based on its unique icao24 number) can have multiple
# entries. If those are not sorted by their icao24 number and timestamp, 
# they will be treated as different aircrafts.
def sort_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts values in df by the columns provided.
    Input:
        df: a Dataframe
        cols: columns that we want to sort the Dataframe with.
    Returns:
        Resulting DataFrame.
    """
    df = df.sort_values(by=['icao24', 'timestamp'])
    return df

# get the number of unique aircrafts present in dataset.
def num_aircrafts(df: pd.DataFrame, col='timestamp'):
    """Outputs the number of unique aircrafts in our dataset.
    """
    return df[col].nunique() 


def initialise_features_and_target() -> tuple[list[str], list[str]]:
    """initialise features and target lists.

    Returns:
        feature and target columns list.
    """
    target = ['delta_latitude', 
              'delta_longitude']
    
    features = ['velocity', 'vertical_rate', 'baro_altitude',
            'delta_time', 'track_sin', 'track_cos',
            'acceleration', 'turn_rate', 'climb_phase',
            'hour_sin', 'hour_cos', 'on_ground', 'cat_Heavy', 
            'cat_Large', 'cat_Light', 'cat_Medium', 
            'cat_Small', 'cat_Super', 'temp_kelvin', 
            'wind_speed', 'wind_dir'
            ]
    
    return features, target

def integrate_weather_data(df_flights, df_weather):
    print("Fusing both the datasets")
    df_flights = df_flights.sort_values('timestamp')
    df_weather = df_weather.sort_values('timestamp')
    
    if df_flights['timestamp'].dt.tz is not None:
        df_flights['timestamp'] = df_flights['timestamp'].dt.tz_localize(None)
    if df_weather['timestamp'].dt.tz is not None:
        df_weather['timestamp'] = df_weather['timestamp'].dt.tz_localize(None)
        
    df_merged = pd.merge(df_flights, df_weather, on='timestamp', how='outer')
    df_merged = df_merged.sort_values('timestamp')
    
    df_merged = df_merged.set_index('timestamp')
    df_merged['wind_speed'] = df_merged['wind_speed'].interpolate(method='time')
    df_merged['wind_dir'] = df_merged['wind_dir'].interpolate(method='time')
    df_merged['temperature'] = df_merged['temperature'].interpolate(method='time')
    
    df_merged = df_merged.reset_index()
    
    df_merged = df_merged.dropna(subset=['icao24'])
    return df_merged