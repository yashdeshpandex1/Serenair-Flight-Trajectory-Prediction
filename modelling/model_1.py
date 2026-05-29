import torch
import torch.nn as nn

class model1(nn.Module):
    """A baseline lstm model._
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(model1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        nn.init.uniform_(self.fc.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc.bias)
        
        
    def forward(self, X):
        """
        X: input of shape (batch_size, seq_length, input_size).
        """
        # LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(X)
        
        # Take the output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Pass through output layer
        output = self.fc(last_output)
        
        return output
