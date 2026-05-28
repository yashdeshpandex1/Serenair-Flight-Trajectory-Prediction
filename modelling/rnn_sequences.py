from tqdm import tqdm
import numpy as np

def create_sequences(flight_X, flight_y, flight_anchor, window_size=10):
    """
    Slide a window over a single flight's data,
    making it appropriate for our future RNN model.
    anchor is for reference to absolute coordinates (
        this is important because we are not predicting 
        absolute but delta coordinates
    )
    """
    X_seq, y_seq, anchor_seq = [], [], []
    
    for i in range(len(flight_X) - window_size):
        X_seq.append(flight_X[i: i + window_size]) # [0:10, 10:20 and so on...]
        y_seq.append(flight_y[i + window_size]) 
        last_step_idx = i + window_size - 1 # 9th index in 0:10
        anchor_seq.append(flight_anchor[last_step_idx])
        
    return np.array(X_seq), np.array(y_seq), np.array(anchor_seq)

def prepare_rnn_data(X, y, anchors, groups, window_size=10):
    """
    Builds up on create_sequences function to further
    create the desired RNN dataset for our models.
    """
    all_X, all_y, all_anchors = [], [], []
    
    # return unique flights for tqdm bar
    unique_flights, start_indices = np.unique(groups, return_index=True) 
    
    # define the start indices
    start_indices = np.append(start_indices, len(groups))
    
    for i in tqdm(range(len(unique_flights)), desc='Extracting Flight Sequences...'):
        start_idx = start_indices[i] # here's the starting point a particular sequence
        end_idx = start_indices[i+1] # ending point of the same
        
        flight_X = X[start_idx: end_idx] # create X seq
        flight_y = y[start_idx: end_idx] # create y seq
        flight_anchor = anchors[start_idx: end_idx] # ref anchor
        
        # apply create_sequences function
        if len(flight_X) > window_size:
            X_s, y_s, anchor_s = create_sequences(flight_X, flight_y, flight_anchor, window_size)
            all_X.append(X_s)
            all_y.append(y_s)
            all_anchors.append(anchor_s)
            
    # return X, y and anchors        
    return np.vstack(all_X), np.vstack(all_y), np.vstack(all_anchors)

def save_sequences(train_df, test_df, features, target, window_size=10):
    
    anchor_cols = ['raw_latitude', 'raw_longitude']
    
    X_train_seq, y_train_seq, anchor_train_seq = prepare_rnn_data(
        X = train_df[features].values,
        y = train_df[target].values,
        anchors = train_df[anchor_cols].values,
        groups = train_df['icao24'].values,
        window_size = window_size
    )
    
    X_test_seq, y_test_seq, anchor_test_seq = prepare_rnn_data(
        X = test_df[features].values,
        y = test_df[target].values,
        anchors = test_df[anchor_cols].values,
        groups = test_df['icao24'].values,
        window_size = window_size
    )
    
    np.savez_compressed('flight_data_for_rnn.npz',
                        X_train = X_train_seq,
                        y_train = y_train_seq,
                        anchor_train = anchor_train_seq,
                        X_test = X_test_seq,
                        y_test = y_test_seq,
                        anchor_test = anchor_test_seq)
    print('Data saved successfully..')

