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
            'hour_sin', 'hour_cos', 'euclidean_speed', 'bearing']
    
    return features, target