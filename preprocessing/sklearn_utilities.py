# All the sklearn utilities and operations

# Imports
import pandas as pd # Dataframe operations
# import GroupShuffleSplit from sklearn
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.preprocessing import StandardScaler # import StandardScaler


# Since our dataset has groups of different aircrafts
# We will have to go for group shuffle split
def group_shuffle_split(df: pd.DataFrame) -> pd.DataFrame:
    """Does a Group Shuffle Split on the dataset.
    Can't use train test split because the training examples are linked in groups.
    """
    
    # create a gss instance
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    # define a training and test set
    training_set, testing_set = next(gss.split(df, groups=df['icao24']))
    df_train = df.iloc[training_set].copy() # X_train, X_test
    df_test = df.iloc[testing_set].copy() # y_train, y_test
    
    return df_train, df_test


# Scale input variables
def scale_dataset(df_train, df_test, features) -> pd.DataFrame:
    """Scales features as well as target variables.
    """
    feature_scaler = StandardScaler() # Create instance of scalar
     
    # fit and transform on training set
    df_train[features] = feature_scaler.fit_transform(df_train[features])
    # only transform on test set
    df_test[features] = feature_scaler.transform(df_test[features])
    
    return df_train, df_test, feature_scaler