import torch
import torch.nn as nn

class HaversineLoss(nn.Module):
    """
    Calculates the mean Haversine distance loss (in meters)
    between the predicted and true coordinate deltas
    """
    def __init__(self, radius=6371000.0):
        super(HaversineLoss, self).__init__()
        # Earth's radius in meters 
        self.R = radius
        
    def forward(self, pred_deltas, true_deltas, anchor_pos):
        # Calculating absolute coordinates from deltas
        pred_pos = anchor_pos + pred_deltas
        true_pos = anchor_pos + true_deltas
        
        # Conversion from degree to radian
        lat1 = torch.deg2rad(pred_pos[:, 0])
        lon1 = torch.deg2rad(pred_pos[:, 1])
        lat2 = torch.deg2rad(true_pos[:, 0])
        lon2 = torch.deg2rad(true_pos[:, 1])
        
        # Vectorized Haversine Loss
        d = torch.sin((lat2-lat1)/2)**2 + \
            torch.cos(lat1) * torch.cos(lat2) * \
                torch.sin((lon2-lon1)/2)**2
                
        
        # Calculate final physical distance and return the mean
        distances = 2 * self.R * torch.asin(torch.sqrt(d))
        return distances.mean()
