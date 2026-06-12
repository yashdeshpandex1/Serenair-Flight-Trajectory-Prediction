import joblib
import torch

def get_unscaled(path='../data/rnn_data/target_scaler.joblib'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_scaler = joblib.load(path)
    target_mean = torch.tensor(target_scaler.center_, dtype=torch.float32)
    target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32)
    return target_mean, target_scale