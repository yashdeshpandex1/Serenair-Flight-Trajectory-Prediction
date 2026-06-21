# Imports
import torch
import torch.nn as nn
from run_experiment import run_experiment
from data_loader import dataloader
from custom_loss_fn import HaversineLoss
from tqdm.auto import tqdm
import joblib
import os
import mlflow
from setup_mlflow import setup_mlflow
from scaler_utilities import get_unscaled
import copy
import argparse 

def evaluate(model, val_loader, eval_criterion,
             target_mean, target_scale, device,
             task='next_instance'):
    """Evaluation module of the engine function.
    """
    criterion = eval_criterion()

    target_mean = target_mean.to(device)
    target_scale = target_scale.to(device)
    
    model.eval()
    val_meters_loss = 0.0
    
    if task == 'next_instance':
        with torch.inference_mode():
            for batch_X, batch_y, batch_anchor in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_anchor = batch_anchor.to(device)
                
                output = model(batch_X)
                pred = output[0] if isinstance(output, tuple) else output
                
                unscaled_pred = (pred * target_scale) + target_mean
                unscaled_y = (batch_y * target_scale) + target_mean
                
                abs_pred = unscaled_pred + batch_anchor
                abs_y = unscaled_y + batch_anchor
                
                val_error = criterion(abs_pred, abs_y)
                val_meters_loss += val_error.item()
                
        return val_meters_loss / len(val_loader)
    
    elif task == 'next_ten_mins':
        with torch.inference_mode():
            for batch_X, batch_y, batch_anchor in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_anchor = batch_anchor.to(device)
                
                output = model(batch_X)
                pred = output[0] if isinstance(output, tuple) else output
                
                pred = pred.view(-1, 10, 2)
                
                unscaled_pred = (pred * target_scale) + target_mean
                unscaled_y = (batch_y * target_scale) + target_mean
                
                abs_pred = unscaled_pred + batch_anchor.unsqueeze(1)
                abs_y = unscaled_y + batch_anchor.unsqueeze(1)
                
                val_error = criterion(abs_pred.view(-1, 2), abs_y.view(-1, 2))
                val_meters_loss += val_error.item()
                
        return val_meters_loss / len(val_loader)


def train(model, train_loader, train_criterion, eval_criterion,
          device, target_mean, target_scale, optimizer,
          task='next_instance'):
    """ Training module for the engine function.
    """
    model.train()
    train_loss_meters = 0.0
    
    if task == 'next_instance':
        for batch_X, batch_y, batch_anchor in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_anchor = batch_anchor.to(device)
            
            output = model(batch_X)
            pred = output[0] if isinstance(output, tuple) else output
            
            loss = train_criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad(): # display haversine loss to understand how off is our model predictions
                unscaled_pred = (pred * target_scale) + target_mean
                unscaled_y = (batch_y * target_scale) + target_mean
                
                abs_pred = unscaled_pred + batch_anchor
                abs_y = unscaled_y + batch_anchor
                
                batch_error = eval_criterion(abs_pred, abs_y)
                train_loss_meters += batch_error.item()
            
        return train_loss_meters/ len(train_loader)
    
    elif task == 'next_ten_mins':
        for batch_X, batch_y, batch_anchor in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_anchor = batch_anchor.to(device)
            
            output = model(batch_X)
            pred = output[0] if isinstance(output, tuple) else output
            pred = pred.view(-1, 10, 2)
            
            loss = train_criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():   # display haversine loss to understand how off is our model predictions
                unscaled_pred = (pred * target_scale) + target_mean
                unscaled_y = (batch_y * target_scale) + target_mean
                
                abs_pred = unscaled_pred + batch_anchor.unsqueeze(1)
                abs_y = unscaled_y + batch_anchor.unsqueeze(1)
                
                batch_error = eval_criterion(abs_pred.view(-1, 2), abs_y.view(-1, 2))
                train_loss_meters += batch_error.item()
                
        return train_loss_meters / len(train_loader)
    
    
def training_loop(model_class, num_epochs=10,
                  train_loss_fn=nn.HuberLoss, eval_loss_fn=HaversineLoss,
                  optimize=torch.optim.Adam, learning_rate=0.001, wd=0.0, 
                  num_layers=2, batch_size=64, hidden_size=64,
                  task='next_instance'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if task == 'next_instance':
        setup_mlflow(task=task)
        _, _, train_loader, val_loader = dataloader(batch_size=batch_size,
                                                    task=task)
        target_mean, target_scale = get_unscaled(task=task)
        target_mean = target_mean.to(device)
        target_scale = target_scale.to(device)
        model = run_experiment(model_name=model_class, num_layers=num_layers,
                               hidden_size=hidden_size, task=task)
    elif task == 'next_ten_mins':
        setup_mlflow(task=task)
        _, _, train_loader, val_loader = dataloader(batch_size=batch_size,
                                                    task=task)
        target_mean, target_scale = get_unscaled(task=task)
        target_mean = target_mean.to(device)
        target_scale = target_scale.to(device)
        model = run_experiment(model_name=model_class, num_layers=num_layers, 
                               hidden_size=hidden_size, task=task)
        
    
    model = model.to(device)
    
    train_criterion = train_loss_fn()
    eval_criterion = eval_loss_fn()
    
    optimizer = optimize(model.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    with mlflow.start_run(run_name=model_class):
        
        mlflow.log_params({
            'model_architecture': model_class,
            'epochs': num_epochs,
            'num_of_layers': num_layers,
            'learning_rate': learning_rate,
            'weight_decay': wd,
            'optimizer': optimize.__name__,
            'train_loss_function': train_loss_fn.__name__
        })
        print(f"Mlflow started tracking {model_class}")
        
        best_val_loss = float('inf')
        best_model_weights = None
        
        epoch_iterator = tqdm(range(num_epochs), desc='Training Progress')
        
        for epoch in epoch_iterator:
            
            avg_train = train(
                model=model,
                train_loader=train_loader,
                train_criterion=train_criterion, 
                eval_criterion=eval_criterion,
                device=device, 
                target_mean=target_mean, 
                target_scale=target_scale, 
                optimizer=optimizer,
                task=task
            )
            
            avg_val = evaluate(
                model=model,
                val_loader=val_loader, 
                eval_criterion=eval_loss_fn,
                target_mean=target_mean, 
                target_scale=target_scale, 
                device=device,
                task=task
            )
            
            scheduler.step(avg_val)
            
            mlflow.log_metrics({
                'train_meters': avg_train,
                'val_haversine_meters': avg_val,
                'learning_rate': optimizer.param_groups[0]['lr']   
            }, step=epoch)
            
            epoch_iterator.set_postfix({
                'Train': f"{avg_train:.2f} meters",
                'Val': f"{avg_val:.2f} meters",
                'LR': f"{optimizer.param_groups[0]['lr']:.6g}"
            })
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_weights = copy.deepcopy(model.state_dict())

                tqdm.write(f"Epoch {epoch} | New Best Val Loss: {best_val_loss:.2f} meters")
        
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        
        if task == 'next_instance':    
            os.makedirs('../models/next_instance', exist_ok=True)   
            local_path = f"../models/next_instance/{model_class}.pth"
            torch.save(model.state_dict(), local_path)
            mlflow.log_artifact(local_path, artifact_path='models')
        elif task == 'next_ten_mins':
            os.makedirs('../models/next_ten_mins', exist_ok=True)   
            local_path = f"../models/next_ten_mins/{model_class}.pth"
            torch.save(model.state_dict(), local_path)
            mlflow.log_artifact(local_path, artifact_path='models')
        
        print(f'Run complete. {model_class} saved securely to MLflow.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    
    parser.add_argument('--model_class', type=str, default='LSTMModelV1', help='Name of the model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay (L2)')
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training and testing set')
    parser.add_argument('--hidden_size', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--task', type=str, default='next_instance', help='which task? (next instance pred / next ten mins pred)')
    
    args = parser.parse_args()
    
    training_loop(
        model_class=args.model_class,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        wd=args.wd,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        task=args.task
    )
