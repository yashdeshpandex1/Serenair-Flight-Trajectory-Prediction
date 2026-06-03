import torch.nn as nn
import torch

class LSTMModelV2(nn.Module):
    
    def __init__(self, input_size,
                 hidden_size, output_size,
                 num_layers=2, dropout_rate=0.2):
        super(LSTMModelV2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X):
        lstm_out, (h_n, c_n) = self.lstm(X)
        
        last_output = lstm_out[:, -1, :]
        
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        
        return output