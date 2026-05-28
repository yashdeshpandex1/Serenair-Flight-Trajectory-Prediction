# Imports
import torch
import torch.nn as nn
from run_experiment import run_experiment
from data_loader import dataloader
from custom_loss_fn import HaversineLoss
from tqdm.auto import tqdm


def training_loop(num_epochs=10, 
                  loss_fn=HaversineLoss,
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
    criterion = loss_fn()
    # Optimizer to be used (default: Adam)
    optimizer = optimize(model.parameters(), lr=learning_rate, 
                         weight_decay=wd)
    
    
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y, batch_anchor in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_anchor = batch_anchor.to(device)
            
            pred = model(batch_X)
            
            loss = criterion(pred, batch_y, batch_anchor)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        # Testing
        model.eval()
        test_loss = 0.0
        with torch.inference_mode():
            for batch_X, batch_y, batch_anchor in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_anchor = batch_anchor.to(device)
                
                pred = model(batch_X)
                batch_loss = criterion(pred, batch_y, batch_anchor)
                test_loss += batch_loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train:.2f} meters | Test Loss: {avg_test:.2f} meters")


if __name__ == "__main__":
    training_loop()