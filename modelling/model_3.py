import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers=2, dropout_rate=0.2):
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          )