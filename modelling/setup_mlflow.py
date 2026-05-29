import mlflow
from dotenv import load_dotenv
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def setup_mlflow():
    """
    Sets up mlflow connection.
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
    
    # Define experiment name
    experiment_name = "serenair_flight_trajectory_detection"
    mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    setup_mlflow()


