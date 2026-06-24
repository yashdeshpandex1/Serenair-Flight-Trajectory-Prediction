import os,  sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from modelling.stage_model import get_ml_client
from modelling.setup_mlflow import setup_mlflow
from modelling.run_experiment import MODEL_REGISTRY
import mlflow.artifacts
import joblib, torch
from pathlib import Path

def try_load_model(run_id, arch, task):
    model_class = MODEL_REGISTRY.get(arch)
    
    if model_class is None:
        print(f" Unknown Architecure in '{arch} in MODEL_REGISTRY")
        return None
    
    out = 20 if task == 'next_ten_mins' else 2
    artifact_subpaths = ['models/models']
    
    modelling_dir = os.path.join(parent_dir, 'modelling')
    if modelling_dir not in sys.path:
        sys.path.insert(0, modelling_dir)
        
    for subpath in artifact_subpaths:
        try:
            local_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=subpath
            )
            pth_files = list(Path(local_dir).rglob("*.pth"))
            if not pth_files:
                print(f"  No .pth file found under artifact path '{subpath}'")
                continue
            
            pth_path = pth_files[0]
            print(f"  Found weights at: {pth_path}")
            
            loaded = torch.load(pth_path, map_location='cpu', weights_only=False)
            
            if isinstance(loaded, dict):
                model = model_class(input_size=21, hidden_size=64,
                                    num_layers=2, output_size=out)
                model.load_state_dict(loaded)
            else:
                model = loaded
                
            print(f"Successfully loaded {arch} from {pth_path}")
            return model
            
        except Exception as e:
            print(f"Failed subpath '{subpath}': {type(e).__name__}: {e}")
            continue
        
    return None


def try_load_scaler(run_id, task):
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path='models/target_scaler.joblib'
        )
        scaler = joblib.load(local_path)
        print(f"Loaded scaler (joblib artifact) from models/target_scaler.joblib")
        return scaler
    except Exception as e:
        print(f"Could not load scaler from MLflow for run {run_id}: {e}")
        return None

    
def load_mlflow_model_and_scaler(registry_name, target_alias,
                                 task='next_instance'):
    
    ml_client = get_ml_client()
    if not ml_client:
        return None, None, None
    
    try:
        print(f"Searching Azure for '{target_alias}' model in {registry_name}...")
        versions = ml_client.models.list(name=registry_name)
        
        for version in versions:
            if version.tags and version.tags.get('alias') == target_alias:
                run_id = version.tags.get('run_id')
                arch = version.tags.get('architecture')
                
                print(f"Found {target_alias} model: {arch} (Run ID: {run_id})")
                
                model = try_load_model(run_id, arch, task)
                scaler = try_load_scaler(run_id, task)
                
                if model is None:
                    print(f"Found alias but could not load model for run {run_id}")
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
        arch = 'Seq2SeqLSTMV1' if task == 'next_ten_mins' else 'LSTMModelV1' 
        model_class = MODEL_REGISTRY.get(arch)
        
        out = 20 if task == 'next_ten_mins' else 2
        model = model_class(input_size=21, hidden_size=64,
                            num_layers=2, output_size=out)
        
        local_path = Path(__file__).parent.parent / 'models' / task / f'{arch}.pth'
        if local_path.exists():
            model.load_state_dict(torch.load(local_path, map_location='cpu'))
            print(f"Loaded local fallback model: {local_path}")
        else:
            print(f"FAILURE: Local fallback model not found at {local_path}")
            model = None
            
    if scaler is None:
        print(f"Falling back to local Scaler on disk for {task}...")
        
        local_scaler_path = Path(__file__).parent.parent / 'data' / f'rnn_data_{task}' / f'target_scaler_{task}.joblib'
        if local_scaler_path.exists():
            scaler = joblib.load(local_scaler_path)
            print(f"Loaded local fallback scaler: {local_scaler_path}")
        if scaler is None:
            print(f"FAILURE: Local fallback scaler for {task} not found!")
            
    if model:
        model.eval()
        model.to('cpu')
        
    return model, scaler


if __name__ == '__main__':
    initialize_inference_engine()