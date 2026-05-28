import torch 
import torch.nn as nn
from model_1 import model1

MODEL_REGISTRY = {
    'lstm': model1
}

def run_experiment(model_name='lstm', hidden_size=32, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name == 'lstm':
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(input_size=13, hidden_size=hidden_size,
                        output_size=2).to(device)
        
    return model
