import torch
import torch.nn as nn

class LSTMModelV1(nn.Module):
    """A simple one layer LSTM model."""
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers=1):
        super(LSTMModelV1, self).__init__()
        
        # Model specifications
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Fully connected Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(X)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # lstm layer -> fc -> output
        output = self.fc(last_output)
        
        return output