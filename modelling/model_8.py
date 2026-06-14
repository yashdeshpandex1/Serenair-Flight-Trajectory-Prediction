import torch
import torch.nn as nn

class HybridConvLSTMBidirectionalModel(nn.Module):
    """A Hybrid (Conv + Bidirectional LSTM) Model with 
        two fully connected layers and a skip connection."""
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers, dropout_rate=0.2):
        super(HybridConvLSTMBidirectionalModel, self).__init__()
        
        # Conv Layer
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.relu = nn.ReLU() # relu activation
        self.dropout = nn.Dropout(dropout_rate) # dropout
        self.skip_projector = nn.Linear(input_size, 32) # skip connection
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)   # fully connected layer 1
        self.fc2 = nn.Linear(64, output_size)   # fully connected layer 2
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # permute X shape to (batch_size, input_size, seq_length)
        x_cnn = X.permute(0, 2, 1)
        x_cnn = self.relu(self.conv1(x_cnn))    # apply relu activation
        
        # permute X shape back to (batch_size, seq_length, input_size)
        x_rnn = x_cnn.permute(0, 2, 1)
        skip = self.skip_projector(X)   # skip connection
        x_combined = self.dropout(x_rnn + skip) # combine skip connection & x_rnn
        
        # LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, _) = self.lstm(x_combined)
        
        # Take mature outputs from both forward and backward pass (bidirectional)
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        
        # Concat forward and backward output to get the output
        last_output = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Conv Layer -> relu -> skip connection -> 
        # Bidirectional LSTM Layers * 2 -> fc1 -> relu -> dropout -> 
        # fc2 -> output
        x = self.dropout(self.relu(self.fc1(last_output)))
        output = self.fc2(x)
        
        return output