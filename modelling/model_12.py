import torch
import torch.nn as nn

class Seq2SeqTrajectoryLSTMV2(nn.Module):
    
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers):
        super(Seq2SeqTrajectoryLSTMV2, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_steps = output_size // 2
        
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        
        self.decoder = nn.LSTM(input_size=2 + hidden_size,
                               hidden_size = hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, X):
        batch_size = X.size(0)
        
        _, (h_n, c_n) = self.encoder(X)
        
        physics_context = h_n[-1].unsqueeze(1)
        outputs = []
        
        prev_coord = torch.zeros(batch_size, 1, 2, device=X.device)
        
        for _ in range(self.future_steps):
            decoder_input = torch.cat([prev_coord, physics_context], dim=2)
            
            out, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            
            step_pred = self.fc(out)
            outputs.append(step_pred)
            
            prev_coord = step_pred
            
        final_trajectory = torch.cat(outputs, dim=1)
        return final_trajectory.view(batch_size, -1)