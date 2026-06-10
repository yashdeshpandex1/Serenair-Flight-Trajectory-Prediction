import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """A Two Layered GRU model with two fully connected layers."""
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers=2, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        
        # GRU Layer
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        
        self.fc1 = nn.Linear(hidden_size, 64)   # fully connected layer 1
        self.fc2 = nn.Linear(64, output_size)   # fully connected layer 2
        
        self.relu = nn.ReLU()   # ReLU activation 
        self.dropout = nn.Dropout(dropout_rate) # dropout
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # GRU returns (output, h_n)
        gru_out, h_n = self.gru(X)
        
        # Take the output from the last time step
        last_output = gru_out[:, -1, :]
        
        # lstm layers -> fc1 -> relu -> dropout -> fc2 -> output
        x = self.dropout(self.relu(self.fc1(last_output)))
        output = self.fc2(x)
        
        return output