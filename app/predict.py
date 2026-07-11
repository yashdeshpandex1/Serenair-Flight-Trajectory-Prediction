import torch

def predict_for_next_instance(X_tensor, plane_metadata, model, scaler):
    if X_tensor.nelement() == 0 or not plane_metadata or model is None:
        return []
    
    with torch.no_grad():
        scaled_preds = model(X_tensor)
        
    scaled_preds_np = scaled_preds.numpy()
    
    if scaler:
        real_deltas = scaler.inverse_transform(scaled_preds_np)
    else:
        real_deltas = scaled_preds_np

    if real_deltas.ndim == 1:
        real_deltas = real_deltas.reshape(-1, 2)
        
    final_trajectories = []
    
    for i, plane in enumerate(plane_metadata):
        current_lat = plane['current_lat']
        current_lon = plane['current_lon']
        
        delta_lat = float(real_deltas[i][0])
        delta_lon = float(real_deltas[i][1])
        
        final_trajectories.append({
            'icao24': plane['icao24'],
            'current_position': {
                'lat': current_lat,
                'lon': current_lon
            },
            'prediction_position': {
                'lat': current_lat + delta_lat,
                'lon': current_lon + delta_lon
            }
        })
        
    return final_trajectories

def predict_for_next_ten_mins(X_tensor, plane_metadata, model, scaler):
    if X_tensor.nelement() == 0 or not plane_metadata or model is None:
        return []
    
    with torch.no_grad():
        scaled_preds = model(X_tensor)
    
    batch_size = scaled_preds.shape[0]
    scaled_preds_np = scaled_preds.numpy().reshape(-1, 2)
    
    if scaler:
        real_deltas_flat = scaler.inverse_transform(scaled_preds_np)
    else:
        real_deltas_flat = scaled_preds_np
    
    real_deltas = real_deltas_flat.reshape(batch_size, 10, 2) 
    
    final_trajectories = []
    
    for i, plane in enumerate(plane_metadata):
        current_lat = plane['current_lat']
        current_lon = plane['current_lon']
        
        full_path = []
        cumulative_lat = current_lat
        cumulative_lon = current_lon
        
        for step in range(10):
            cumulative_lat += float(real_deltas[i][step][0])
            cumulative_lon += float(real_deltas[i][step][1])
            full_path.append({'lat': cumulative_lat, 'lon': cumulative_lon})
        
        final_trajectories.append({
            'icao24': plane['icao24'],
            'current_position': {'lat': current_lat, 'lon': current_lon},
            'prediction_position': {'lat': cumulative_lat, 'lon': cumulative_lon},
            'full_path': full_path
        })
    
    return final_trajectories