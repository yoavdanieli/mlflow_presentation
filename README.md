# MLFlow Demo

## Requirements
- Install mlflow using `pip install mlflow[extras]`
- Install conda
- Install pandas
- Install sqlite3
- Install an FTP server - make the ftp directory ~/ftp

## Tracking Demo
- Run mlflow_tracking_demo.py
- Run mlflow ui

## Running Full MLflow Server
- Start the FTP server
- Run run_mlflow_server.sh

## Training the Model
You can either:
- execute run_models.py directly
- execute it through mlflow project with run_project.sh to reproduce the environment
 You can pass alpha and l1_level parameters.

## Deploying the Model
Use serve_model.sh to start an HTTP server with the model (and run_id).
You can ask the model to predict on example data with test_model_server.sh. You can copy the curl in the file and alter the data.