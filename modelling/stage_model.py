import os
from .setup_mlflow import setup_mlflow
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
import argparse
import json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(_ROOT, 'config.json') 

def get_ml_client():
    try:
        with open(config_path) as f:
            config = json.load(f)

        return MLClient(DefaultAzureCredential(),
                        subscription_id=config['subscription_id'],
                        resource_group_name=config['resource_group'],
                        workspace_name=config['workspace_name'])
    except Exception as e:
        print(f"No Azure config file detected: {e}")
        return None
    

def stage_model_to_registry(task='next_instance', stage=False, top_k=5):
    setup_mlflow(task=task)
    client = MlflowClient()
    
    if task == 'next_instance':
        experiment_name = 'next instance trajectory prediction'
        registry_name = 'Serenair_Next_Instance'
    elif task == 'next_ten_mins':
        experiment_name = 'next ten minutes trajectory prediction'
        registry_name = 'Serenair_Next_Ten_Minutes'
        
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name} not found'")
        return
    
    runs = client.search_runs(
        experiment_ids = [experiment.experiment_id],
        max_results = 50
    )
    
    sorted_runs = sorted(runs,
                         key=lambda r: r.data.metrics.get('best_val_haversine_meters', float('inf')))
    top_runs = sorted_runs[:top_k]
    
    print(f"\n Top {len(top_runs)} Models for '{task.upper()}': \n")
    for i, run in enumerate(top_runs, 1):
        model_name = run.data.params.get('model_architecture', 'Unknown Model')
        val_error = run.data.metrics.get('best_val_haversine_meters', float('inf'))
        best_epoch = run.data.metrics.get('best_epoch', 'N/A')
        print(f"Rank {i}: {model_name} | Error: {val_error:.2f}m | ID: {run.info.run_id}")
    print('-' * 60)
    
    if not stage or not top_runs:
        return 
    
    ml_client = get_ml_client()
    
    try:
        existing_versions = list(ml_client.models.list(name=registry_name))
        registered_run_ids = {m.tags.get('run_id') for m in existing_versions}
    except Exception:
        registered_run_ids = set()
        
        
    aliases_to_register = [
        (top_runs[0], 'Production'),
        (top_runs[1], 'Staging') if len(top_runs) > 1 else None
    ]
    
    for entry in aliases_to_register:
        if entry is None:
            continue
        
        run, alias = entry
        run_id = run.info.run_id
        arch = run.data.params.get('model_architecture', 'Unknown')
        best_val_error = run.data.metrics.get('best_val_haversine_meters', 0)
        best_epoch = run.data.metrics.get('best_epoch', 'N/A')
        
        if run_id in registered_run_ids:
            print(f"[{alias}] {arch} (run: {run_id}) already registered, skipping.")
            continue

        model_uri = f"runs:/{run_id}/models"
        
        try:
            az_model = Model(
                path=model_uri,
                name=registry_name,
                description=f"{arch} | {alias} | Best Error: {best_val_error:.2f}m @ Epoch {best_epoch}",
                type=AssetTypes.MLFLOW_MODEL,
                tags={
                    'alias': alias,
                    'architecture': arch,
                    'run_id': run_id,
                    'task': task,
                    'best_val_haversine_meters': str(best_val_error),
                    'best_epoch': str(best_epoch)
                }
            )
            
            registered = ml_client.models.create_or_update(az_model)
            print(f"[{alias}] Registered {arch} (run: {run_id}) as version {registered.version}")
        except Exception as e:
            print(f"Failed to register {alias} model: {e}")
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['next_instance', 'next_ten_mins'])
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--stage', action='store_true')
    args = parser.parse_args()
    
    stage_model_to_registry(task=args.task, top_k=args.top_k, stage=args.stage)