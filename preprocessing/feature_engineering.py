# Feature engineering for our dataset

# Imports
import pandas as pd # for working with dataframes
import numpy as np # for mathematical operations


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new flight features from existing ones.
    """
    df = df.copy() # Create a copy
    
    df['timestamp'] = pd.to_datetime(df['timestamp']) # convert to datetime type
    
    df['delta_time'] = ( # create delta time
        df.groupby('icao24')['timestamp'] # groupby icao24 and timestamp
        .diff() # take diff for every timestamp for a specific plane
        .dt.total_seconds() # convert to total seconds
        .fillna(0) # fill the first entries with 0
    )
    
    theta = np.deg2rad(df['true_track']) # convert degrees to radians
    df['track_sin'] = np.sin(theta) # Create sin track.
    df['track_cos'] = np.cos(theta) # Create cos track.
    
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    delta_zero = df['delta_time'].replace(0, 1) # replace delta 0 with 1.
    
    # Calculate previous velocity and acceleration from previous velocity.
    df['prev_velocity'] = (df.groupby('icao24')['velocity'].shift(1)).fillna(0)
    df['acceleration'] = ((df['velocity'] - df['prev_velocity']) / delta_zero).fillna(0)
    
    # Calculate previous track
    df['prev_track'] = (df.groupby('icao24')['true_track'].shift(1)).fillna(0)
    track_diff = df['true_track'] - df['prev_track'] # Calculate track difference.
    track_diff = (track_diff + 180) % 360 - 180
    # finally calculate turn rate of aircraft
    df['turn_rate'] = (track_diff / delta_zero).fillna(0)
    
    # Initialise a climb phase for our aircraft
    df['climb_phase'] = 0 # stable
    df.loc[df['vertical_rate'] > 1.0, 'climb_phase'] = 1 # ascending
    df.loc[df['vertical_rate'] < -1.0, 'climb_phase'] = -1 # descending
    
    # Calculate delta latitude and longitude from absolute latitude and longitude
    # This will be our target variables
    df['delta_latitude'] = df.groupby('icao24')['latitude'].diff().fillna(0)
    df['delta_longitude'] = df.groupby('icao24')['longitude'].diff().fillna(0)
    
    # Transponder gap > 60 seconds for consistency
    df = df[df['delta_time'] <= 60]
    df = df.dropna(subset=['delta_latitude', 'delta_longitude', 'delta_time'])
    
    return df
