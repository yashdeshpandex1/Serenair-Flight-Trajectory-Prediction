import torch
import torch.nn as nn

class HybridConvLSTMModelV1(nn.Module):
    """A Hybrid (Convolutional + RNN) Model with two fully connected layers."""
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers=2, dropout_rate=0.2):
        super(HybridConvLSTMModelV1, self).__init__()
        
        # Conv Layer
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.relu = nn.ReLU()   # relu activation
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        
        self.fc1 = nn.Linear(hidden_size, 64)   # fully connected layer 1
        self.fc2 = nn.Linear(64, output_size)   # fully connected layer 2
        
        self.dropout = nn.Dropout(dropout_rate) # dropout 
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # permute X shape to (batch_size, input_size, seq_length) 
        x_cnn = X.permute(0, 2, 1)
        x_cnn = self.relu(self.conv1(x_cnn))    # apply relu activation
        
        # permute X shape back to (batch_size, seq_length, input_size)
        x_rnn = x_cnn.permute(0, 2, 1)
        # LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(x_rnn)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # conv layer -> relu -> LSTM layer -> fc1 -> relu -> dropout -> fc2 -> output
        x = self.dropout(self.relu(self.fc1(last_output)))
        output = self.fc2(x)
        
        return output
        