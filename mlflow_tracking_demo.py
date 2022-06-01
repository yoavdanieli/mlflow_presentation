import sys
import time
import mlflow
from mlflow.tracking import MlflowClient

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('Enter interval and multiplier')
    exit(1)

  interval = float(sys.argv[1])
  multiplier = int(sys.argv[2])

  with mlflow.start_run() as run:
    print(f'Starting run {run.info.run_id}')

    # # Create model
    # if not mlflow_client.get_registered_model(MODEL_NAME):
    #   mlflow_client.create_registered_model()

    # if not mlflow_client.get_model_version(MODEL_NAME, model_version):
    #   mlflow_client.create_model_version(MODEL_NAME, model_version)

    mlflow.log_param('interval', interval)
    mlflow.log_param('multiplier', multiplier)

    value = 1

    for _ in range(5):
      print(f'logging value {value}')
      mlflow.log_metric('value', value)
      time.sleep(interval)
      value *= multiplier
