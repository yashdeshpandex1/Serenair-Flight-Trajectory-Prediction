import mlflow
from dotenv import load_dotenv
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def setup_mlflow(task='next_instance'):
    """Sets up mlflow connection.

    Args:
        task (str, optional): task name to set up experiment name.
        Defaults to 'next_instance'.
    """
    
    # Try connecting to mlflow on azure
    try:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print("Connected to azure mlflow server")
    # If there's no credentials then setup a local server
    except:
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        print("Connected to local sqlite server")
    
    if task=='next_instance':
        # Define experiment name
        experiment_name = "next instance trajectory prediction"
        mlflow.set_experiment(experiment_name)
    elif task == 'next_ten_mins':
        experiment_name = "next ten minutes trajectory prediction"
        mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    setup_mlflow()


