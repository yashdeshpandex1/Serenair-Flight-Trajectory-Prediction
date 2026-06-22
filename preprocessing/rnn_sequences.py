from tqdm import tqdm
import numpy as np
from pathlib import Path

def create_sequences(flight_X, flight_y, flight_anchor, window_size=10,
                     task='next_instance'):
    """
    Slide a window over a single flight's data,
    making it appropriate for our future RNN models.
    anchor is for reference to absolute coordinates (
        this is important because we are not predicting 
        absolute but delta coordinates
    )
    """
    X_seq, y_seq, anchor_seq = [], [], []
    if task == 'next_instance':
        for i in range(len(flight_X) - window_size):    # sliding window
            X_seq.append(flight_X[i: i + window_size]) # [0:10, 1:11, 2:12 and so on...]
            y_seq.append(flight_y[i + window_size]) 
            last_step_idx = i + window_size - 1 # 9th index in 0:10
            anchor_seq.append(flight_anchor[last_step_idx]) # add the last coordinate to anchor 

        return np.array(X_seq), np.array(y_seq), np.array(anchor_seq)
    
    elif task == 'next_ten_mins':
        steps_ahead = 60    # skip 60 timesteps
        
        if len(flight_X) < (window_size + steps_ahead): # if len is less than 70, dump it
            return np.array([]), np.array([]), np.array([])    
        
        for i in range(len(flight_X) - window_size - steps_ahead + 1):  #sliding window
            X_seq.append(flight_X[i: i + window_size])  # [0:10, 1:11, 2:12 and so on...]
            last_step_idx = i + window_size -1 # 9th index in 0:10
            future_targets = [] # make a list of y (10 instances of (lat, lon))
            for minute in range(1, 11):
                target_idx = last_step_idx + (minute * 6)
                future_targets.append(flight_y[target_idx]) # add to the future targets list
                
            y_seq.append(future_targets)
            anchor_seq.append(flight_anchor[last_step_idx])
            
        return np.array(X_seq), np.array(y_seq), np.array(anchor_seq)


def prepare_rnn_data(df, features, target, anchor_cols, window_size=10,
                     task='next_instance'):
    """
    Builds up on create_sequences function to further
    create the desired RNN dataset for our models.
    """
    all_X, all_y, all_anchors = [], [], []
    
    if task == 'next_instance':
        print(f"Creating sequences for next instance prediction.")
        for _, flight_data in tqdm(df.groupby('icao24'), desc='Extracting Flight Sequences (next_instance)'):
            flight_data = flight_data.sort_values(by='timestamp')

            flight_X = flight_data[features].values
            flight_y = flight_data[target].values
            flight_anchor = flight_data[anchor_cols].values

            if len(flight_X) > window_size:
                X_s, y_s, anchor_s = create_sequences(flight_X, flight_y, flight_anchor, window_size=window_size, task='next_instance')
                
                if X_s.size > 0:
                    all_X.append(X_s)
                    all_y.append(y_s)
                    all_anchors.append(anchor_s)
                    
        if not all_X:
            return np.array([]), np.array([]), np.array([])
        
        # return X, y and anchors        
        return np.vstack(all_X), np.vstack(all_y), np.vstack(all_anchors)
    
    elif task == 'next_ten_mins':
        print(f"Creating sequences for next ten minutes prediction.")
        for _, flight_data in tqdm(df.groupby('icao24'), desc='Extracting Flight Sequences (next_ten_mins)'):
            
            flight_data = flight_data.sort_values(by='timestamp')
            
            flight_X = flight_data[features].values
            flight_y = flight_data[target].values
            flight_anchor = flight_data[anchor_cols].values
            
            X_s, y_s, anchor_s = create_sequences(flight_X, flight_y, flight_anchor, window_size=10, task='next_ten_mins')
            
            if X_s.size > 0:
                all_X.append(X_s)
                all_y.append(y_s)
                all_anchors.append(anchor_s)
                
        if not all_X:
            return np.array([]), np.array([]), np.array([])
        
        return np.vstack(all_X), np.vstack(all_y), np.vstack(all_anchors)
        

def save_sequences(train_df, test_df, features, target, window_size=10,
                   task='next_instance'):
    
    anchor_cols = ['raw_latitude', 'raw_longitude']
    X_train_seq, y_train_seq, anchor_train_seq = prepare_rnn_data(
        df=train_df,
        features=features,
        target=target,
        anchor_cols=anchor_cols,
        window_size=window_size,
        task=task
    )
    
    X_test_seq, y_test_seq, anchor_test_seq = prepare_rnn_data(
        df=test_df,
        features=features,
        target=target,
        anchor_cols=anchor_cols,
        window_size=window_size,
        task=task
    )
    
    data_dir = Path(f'../data/rnn_data_{task}/')
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / 'flight_data_for_rnn_{task}.npz'
    
    np.savez_compressed(save_path,
                        X_train = X_train_seq,
                        y_train = y_train_seq,
                        anchor_train = anchor_train_seq,
                        X_test = X_test_seq,
                        y_test = y_test_seq,
                        anchor_test = anchor_test_seq)
    print(f'Data saved successfully ({task})..')