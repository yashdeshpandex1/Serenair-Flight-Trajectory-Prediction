# Imports
import torch
import torch.nn as nn
from run_experiment import run_experiment
from data_loader import dataloader
from custom_loss_fn import HaversineLoss
from tqdm.auto import tqdm
import joblib


def training_loop(num_epochs=10,
                  train_loss_fn=nn.MSELoss, 
                  eval_loss_fn=HaversineLoss,
                  optimize=torch.optim.Adam, 
                  learning_rate=0.001,
                  wd=1e-5):
    """performs training and testing on the model provided in run_experiment.

    Args:
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        loss_fn (_type_, optional): Loss function to be selected. Defaults to HaversineLoss.
        optimize (_type_, optional): Optimizer to be selected. Defaults to torch.optim.Adam.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.001.
        wd (_type_, optional): weight decay. Defaults to 1e-5.
    """
    
    # Load train_loader and test loader
    _, _, train_loader, test_loader = dataloader()
    # Set the default device to be cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use the model provided in run_experiment
    model = run_experiment()
    # Use the loss function (Haversine Loss is the default)
    train_criterion = train_loss_fn()
    eval_criterion = eval_loss_fn()
    
    # Optimizer to be used (default: Adam)
    optimizer = optimize(model.parameters(), lr=learning_rate, 
                         weight_decay=wd)
    
    #
    target_scaler = joblib.load('../data/rnn_data/target_scaler.joblib')
    target_mean = torch.tensor(target_scaler.center_, dtype=torch.float32).to(device)
    target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32).to(device)
    
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_meters_loss = 0.0
        
        # Unpack batch_X, batch_y, batch_anchor and send to 'device' batch-wise
        for batch_X, batch_y, batch_anchor in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_anchor = batch_anchor.to(device)
            
            # Make predictions
            pred = model(batch_X)
            
            # Calculate loss
            train_loss = train_criterion(pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                unscaled_pred = (pred * target_scale) + target_mean
                unscaled_y = (batch_y * target_scale) + target_mean
                abs_pred = unscaled_pred + batch_anchor
                abs_y = unscaled_y + batch_anchor
                
                training_error = eval_criterion(abs_pred, abs_y)
                train_meters_loss += training_error.item()

            
        # Testing
        model.eval()
        test_meters_loss = 0.0
        with torch.inference_mode():
            # Unpack batch_X, batch_y and batch_anchor and send to 'device' batch-wise
            for batch_X, batch_y, batch_anchor in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_anchor = batch_anchor.to(device)
                
                # Make predictions, calculate loss and and sum them up.
                pred = model(batch_X)
                
                unscaled_pred = (pred * target_scale) + target_mean
                unscaled_y = (batch_y * target_scale) + target_mean
                abs_pred = unscaled_pred + batch_anchor
                abs_y = unscaled_y + batch_anchor
                
                test_error = eval_criterion(abs_pred, abs_y)
                test_meters_loss += test_error.item()
          
        # Finally, calculate average training loss and testing loss      
        avg_train = train_meters_loss / len(train_loader)
        avg_test = test_meters_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train:.2f} meters | Test Loss: {avg_test:.2f} meters")


if __name__ == "__main__":
    training_loop()