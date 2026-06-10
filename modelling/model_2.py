import torch.nn as nn
import torch

class LSTMModelV2(nn.Module):
    """A Two Layered LSTM model with two fully connected layers."""
    def __init__(self, input_size,
                 hidden_size, output_size,
                 num_layers=2, dropout_rate=0.2):
        super(LSTMModelV2, self).__init__()
        
        # Model Specifications
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        
        self.fc1 = nn.Linear(hidden_size, 64)   # first fully connected layer
        self.fc2 = nn.Linear(64, output_size)   # second fully connected layer
        self.relu = nn.ReLU()    # ReLU activation
        self.dropout = nn.Dropout(dropout_rate)  # dropout
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(X)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # lstm layers -> fc1 -> relu -> dropout -> fc2 -> output
        x = self.dropout(self.relu(self.fc1(last_output)))
        output = self.fc2(x)
        
        return output