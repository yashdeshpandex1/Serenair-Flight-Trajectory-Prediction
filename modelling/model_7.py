import torch
import torch.nn as nn

class HybridConvLSTMV3(nn.Module):
    """A Hybrid Conv + LSTM Layer model with two fully connected
       layer and skip connection."""
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers, dropout_rate=0.2):
        super(HybridConvLSTMV3, self).__init__()
        
        # Conv1 Layer
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        # Conv2 Layer
        self.conv2 = nn.Conv1d(in_channels=32, 
                               out_channels=32,
                               kernel_size=3, 
                               padding=1)
        self.leaky_relu = nn.LeakyReLU() # Leaky relu activation
        
        self.spatial_dropout = nn.Dropout1d(dropout_rate)   # Use spatial dropout for conv layers
        self.dropout = nn.Dropout(dropout_rate)     # standard dropout for fc layers
        self.skip_projector = nn.Linear(input_size, 32) # skip connection
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        
        self.fc1 = nn.Linear(hidden_size, 64) # fully connected layer 1
        self.fc2 = nn.Linear(64, output_size) # fully connected layer 2
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # permute X shape to (batch_size, input_size, seq_length)
        x_cnn = X.permute(0, 2, 1)
        x_cnn = self.leaky_relu(self.conv1(x_cnn)) # Conv layer and leaky relu activation
        x_cnn = self.spatial_dropout(x_cnn) # Spatial dropout for conv layer
        x_cnn = self.leaky_relu(self.conv2(x_cnn)) # Conv layer and leaky relu 
        
        # permute X shape back to (batch_size, seq_length, input_size)
        x_rnn = x_cnn.permute(0, 2, 1)
        skip = self.skip_projector(X)   # skip connection
        x_combined = self.dropout(x_rnn + skip) # combine skip connection & x_rnn
        
        # LSTM returns (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x_combined)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Conv Layer 1 -> leaky relu -> dropout -> conv 2 layer -> leaky relu ->
        # skip connection -> LSTM Layers * 2 -> fc1 -> leaky relu -> dropout ->
        # fc2 -> output 
        x = self.dropout(self.leaky_relu(self.fc1(last_output)))
        output = self.fc2(x)
        
        return output 