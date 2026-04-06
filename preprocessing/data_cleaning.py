# Cleans the data gathered by getting_data.ipynb

# Imports
import numpy as np
import pandas as pd

# Clean the dataframe
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the OpenSky Dataset with most appropriate methods.
    """
    df = df.dropna(subset=['latitude', 'longitude']) # drop the rows where latitude and longitude are missing
    df.loc[:, 'baro_altitude'] = df['baro_altitude'].fillna(df['baro_altitude'].median()) # fill baro altitude missing values with median
    df.loc[:, 'velocity'] = df['velocity'].fillna(df['velocity'].median()) # fill missing velocity with median as well
    df.loc[:, 'vertical_rate'] = df['vertical_rate'].fillna(value=0) # fill vertical rate with 0s
    df = df.drop(columns=['callsign', 'geo_altitude']) # we don't really need these columns for our application
    return df