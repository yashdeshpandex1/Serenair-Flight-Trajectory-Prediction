# Imports
import torch
import torch.nn as nn
from run_experiment import run_experiment
from data_loader import dataloader
from custom_loss_fn import HaversineLoss
from tqdm.auto import tqdm
import joblib
import mlflow
from setup_mlflow import setup_mlflow
from scaler_utilities import get_unscaled
import copy


def evaluate(model, val_loader, eval_criterion,
             target_mean, target_scale, device):
    
    criterion = eval_criterion()

    target_mean = target_mean.to(device)
    target_scale = target_scale.to(device)
    
    model.eval()
    val_meters_loss = 0.0
    
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


def train(model, train_loader, train_criterion, eval_criterion,
          device, target_mean, target_scale, optimizer):
    """"""
    model.train()
    train_loss_meters = 0.0
    
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
        
        with torch.no_grad():
            unscaled_pred = (pred * target_scale) + target_mean
            unscaled_y = (batch_y * target_scale) + target_mean
            
            abs_pred = unscaled_pred + batch_anchor
            abs_y = unscaled_y + batch_anchor
            
            batch_error = eval_criterion(abs_pred, abs_y)
            train_loss_meters += batch_error.item()
        
    return train_loss_meters/ len(train_loader)

    
def training_loop(model_class, num_epochs=10,
                  train_loss_fn=nn.MSELoss, eval_loss_fn=HaversineLoss,
                  optimize=torch.optim.Adam, learning_rate=0.0001, wd=0.0):
    
    setup_mlflow()
    _, _, train_loader, val_loader = dataloader()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    target_mean, target_scale = get_unscaled()
    target_mean = target_mean.to(device)
    target_scale = target_scale.to(device)
    
    model = run_experiment(model_name=model_class)
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
            'learning_rate': learning_rate,
            'weight_decay': wd,
            'optimizer': optimize.__name__,
            'train_loss_function': train_loss_fn.__name__
        })
        print(f"Mlflow started tracking {model_class}")
        
        best_val_loss = float('inf')
        best_model_weights = None
        
        for epoch in tqdm(range(num_epochs), desc='Training Progress'):
            
            avg_train = train(
                model=model,
                train_loader=train_loader,
                train_criterion=train_criterion, 
                eval_criterion=eval_criterion,
                device=device, 
                target_mean=target_mean, 
                target_scale=target_scale, 
                optimizer=optimizer
            )
            
            avg_val = evaluate(
                model=model,
                val_loader=val_loader, 
                eval_criterion=eval_loss_fn,
                target_mean=target_mean, 
                target_scale=target_scale, 
                device=device
            )
            
            scheduler.step(avg_val)
            
            mlflow.log_metrics({
                'train_meters': avg_train,
                'val_haversine_meters': avg_val,
                'learning_rate': optimizer.param_groups[0]['lr']   
            }, step=epoch)
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_weights = copy.deepcopy(model.state_dict())
        
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
            
        mlflow.pytorch.log_model(model, artifact_path='models',
                                 registered_model_name=model_class)
        
        print(f'Run complete. {model_class} saved securely to MLflow.')
            
            
