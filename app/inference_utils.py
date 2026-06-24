from modelling.stage_model import get_ml_client
from modelling.setup_mlflow import setup_mlflow
from modelling.run_experiment import MODEL_REGISTRY
import mlflow.pytorch, mlflow.sklearn, mlflow.artifacts
import joblib, torch
from pathlib import Path


def load_mlflow_model_and_scaler(registry_name, target_alias, 
                                 task='next_instance'):
    
    ml_client = get_ml_client()
    if not ml_client:
        return None, None, None
    
    try:
        print(f"Searching Azure for '{target_alias}' model in {registry_name}...")
        versions = ml_client.models.list(name=registry_name)
        
        for version in versions:
            if version.tags.get('alias') == target_alias:
                run_id = version.tags.get('run_id')
                arch = version.tags.get('architecture')
                
                print(f"Found {target_alias} model: {arch} (Run ID: {run_id})")
                
                # Load the model
                model = None
                model_paths = [
                    f"runs:/{run_id}/models/models",
                    f"runs:/{run_id}/models"
                    ]
                for m_path in model_paths:
                    try:
                        model = mlflow.pytorch.load_model(m_path,
                                                          map_location='cpu')
                        print(f"Successfully loaded Pytorch Model from {m_path}")
                        break
                    except Exception:
                        continue
                
                if model is None:
                    print(f"Failed to load Pytorch Model from Mlflow for Run ID: {run_id}")
                    continue
                
                scaler = None
                try:
                    scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/target_scaler")
                    print(f"Loaded MLflow Scaler for {task}")
                except Exception:
                    try:
                        try_paths = [
                            f"runs:/{run_id}/models/target_scaler.joblib",
                            f"runs:/{run_id}/scalers/target_scaler_{task}.joblib",
                            f"runs:/{run_id}/artifacts/target_scaler.joblib",
                            f"runs:/{run_id}/target_scaler.joblib"
                        ]
                        for s_path in try_paths:
                            try:
                                artifact_path = mlflow.artifacts.download_artifacts(s_path)
                                scaler = joblib.load(artifact_path)
                                print(f"Loaded MLflow Scaler (Artifact) from {s_path}")
                                break
                            except Exception:
                                continue
                    except Exception as e:
                        print(f"Could not fetch MLflow scaler.")
                return model, scaler, arch
        print(f"No model tagged as '{target_alias}' found")
        return None, None, None
    
    except Exception as e:
        print(f"Error establising communication with Azure ML: {e}")
        return None, None, None
    
    
def initialize_inference_engine(task='next_instance'):
    
    setup_mlflow(task=task)
    registry_name = 'Serenair_Next_Ten_Minutes' if task == 'next_ten_mins' else 'Serenair_Next_Instance'
    
    # 1. Production Model
    model, scaler, arch = load_mlflow_model_and_scaler(registry_name, 
                                                       'Production',
                                                       task)
    
    # 2. Staging Model
    if model is None:
        model, scaler, arch = load_mlflow_model_and_scaler(registry_name, 
                                                           'Staging',
                                                           task)
    
    # 3. Local Fallback Model
    if model is None:
        print(f"Falling back to local Model on disk for {task}...")
        arch = 'Seq2SeqTrajectoryLSTMV1' if task == 'next_ten_mins' else 'LSTMModelV1' 
        model_class = MODEL_REGISTRY.get(arch)
        
        out = 20 if task == 'next_ten_mins' else 2
        model = model_class(input_size=21, hidden_size=64,
                            num_layers=2, output_size=out)
        
        local_path = Path(f'../models/{task}/{arch}.pth')
        
        if local_path.exists():
            model.load_state_dict(torch.load(local_path, map_location='cpu'))
            print(f"Loaded Local fallback model: {local_path}")
        else:
            print(f"FAILURE: Local fallback {local_path} not found!")
            
    if scaler is None:
        print(f"Falling back to local Scaler on disk for {task}...")
        
        local_scaler_path = Path(f"../data/rnn_data_{task}/target_scaler_{task}.joblib")
        if local_scaler_path.exists():
            scaler = joblib.load(local_scaler_path)
            print(f"Loaded local fallback scaler: {local_scaler_path}")
        if scaler is None:
            print(f"FAILURE: Local fallback scaler for {task} not found!")
            
    if model:
        model.eval()
        model.to('cpu')
        
    return model, scaler