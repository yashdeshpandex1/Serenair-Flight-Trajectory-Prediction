import joblib
import torch

def get_unscaled(task='next_instance'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if task == 'next_instance':   
        target_scaler = joblib.load('../data/rnn_data_next_instance/target_scaler_next_instance.joblib')
    elif task == 'next_ten_mins':
        target_scaler = joblib.load('../data/rnn_data_next_ten_mins/target_scaler_next_ten_mins.joblib')
    target_mean = torch.tensor(target_scaler.center_, dtype=torch.float32)
    target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32)
    return target_mean, target_scale