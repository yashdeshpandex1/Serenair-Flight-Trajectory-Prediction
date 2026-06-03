import torch 
import torch.nn as nn
from model_1 import model1
from model_2 import LSTMModelV2

MODEL_REGISTRY = {
    'lstmV1': model1,
    'lstmV2': LSTMModelV2
}

def run_experiment(model_name='lstmV2', hidden_size=32, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name == 'lstmV1':
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(input_size=11, hidden_size=hidden_size,
                        output_size=2).to(device)
    
    if model_name == 'lstmV2':
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(input_size=11, hidden_size=128,
                            output_size=2, num_layers=3).to(device)
        
    return model
