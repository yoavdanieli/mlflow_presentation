import mlflow
from mlflow.tracking import MlflowClient
from wine_model.model import WineModel

run_models = [
    WineModel(),
]

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow_client = MlflowClient()

if __name__ == '__main__':
    for curr_model in run_models:
        with mlflow.start_run() as run:
            curr_model.init()
            curr_model.run(run, mlflow_client)
