# Utilities needed to transform our data gathered by getting_data.ipynb

# Imports
import numpy as np
import pandas as pd

df = pd.read_csv('../data/opensky_raw.csv') # Make a dataframe of our data.

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
def num_aircrafts(df=df, col='timestamp'):
    """Outputs the number of unique aircrafts in our dataset.
    """
    return df[col].nunique()