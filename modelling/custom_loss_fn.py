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
        
    def forward(self, abs_pred, abs_y):
        lat1 = torch.deg2rad(abs_pred[:, 0])
        lon1 = torch.deg2rad(abs_pred[:, 1])
        lat2 = torch.deg2rad(abs_y[:, 0])
        lon2 = torch.deg2rad(abs_y[:, 1])

        d = torch.sin((lat2 - lat1) / 2)**2 + \
            torch.cos(lat1) * torch.cos(lat2) * \
                torch.sin((lon2 - lon1) / 2)**2
                
        distances = 2 * self.R * torch.asin(torch.sqrt(d))
        return distances.mean()