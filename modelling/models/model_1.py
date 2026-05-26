import torch
import torch.nn as nn

class model1(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(model1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        nn.init.uniform_(self.fc.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc.bias)
        
        
    def forward(self, X):
        lstm_out, (h_n, c_n) = self.lstm(X)
        
        last_output = lstm_out[:, -1, :]
        
        output = self.fc(last_output)
        
        return output