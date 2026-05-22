import mlflow
from dotenv import load_dotenv
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def setup_mlflow():
    try:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print("Connected to azure mlflow server")
    except:
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        print("Connected to local sqlite server")
    
    experiment_name = "serenair_flight_trajectory_detection"
    mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    setup_mlflow()


