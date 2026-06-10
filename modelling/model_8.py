import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """Attention block to get context vector and attention weights."""
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention_scorer = nn.Linear(hidden_size, 1)
        
    def forward(self, rnn_output):
        # get attention scores from rnn output
        scores = self.attention_scorer(rnn_output)
        
        # use softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1)
        
        # multiply weights with rnn outputs to get context vectors
        context_vector = torch.sum(attn_weights * rnn_output, dim=1)
        
        # return context vector and attention weights
        return context_vector, attn_weights
    
class AttentionModel(nn.Module):
    """An Attention model with two GRU Layers and two fully connected layers."""
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers=2, dropout_rate=0.2):
        super(AttentionModel, self).__init__()
        
        # GRU Layer
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.relu = nn.ReLU() # relu activation
        self.dropout = nn.Dropout(dropout_rate) # dropout
        
        # Attention Layer
        self.attention = TemporalAttention(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, 64) # fully connected layer 1
        self.fc2 = nn.Linear(64, output_size) # fully connected layer 2
        
    def forward(self, X):
        """X: input of shape (batch_size, seq_length, input_size)."""
        
        # GRU returns (output, h_n)
        gru_out, _ = self.gru(X)
        
        # Get context vector and attention weights from GRU Layer
        context_vector, attn_weights = self.attention(gru_out)
        
        # GRU Layers * 2 -> attention layer -> fc1 -> relu -> dropout ->
        # fc2 -> output
        x = self.dropout(self.relu(self.fc1(context_vector)))
        output = self.fc2(x)
        
        # return output and attention weights
        return output, attn_weights