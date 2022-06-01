# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

from inspect import signature
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class WineModel:
    MODEL_NAME = "WineModel"
    
    def _eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    def init(self):
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        # Read the wine-quality csv file from the URL
        csv_url = (
            "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        )
        try:
            data = pd.read_csv(csv_url, sep=";")
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        self.train_x = train.drop(["quality"], axis=1)
        self.test_x = test.drop(["quality"], axis=1)
        self.train_y = train[["quality"]]
        self.test_y = test[["quality"]]

        self.alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        self.l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    def run(self, run, mlflow_client):
        print(f"Starting run {run.info.run_id}")

        #mlflow_client.create_model_version(self.MODEL_NAME, "model")

        lr = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=42)
        lr.fit(self.train_x, self.train_y)

        predicted_qualities = lr.predict(self.test_x)

        (rmse, mae, r2) = self._eval_metrics(self.test_y, predicted_qualities)
        model_signature = infer_signature(self.train_x, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (self.alpha, self.l1_ratio))
        print("  RMSE: %s" % rmse)  # Root Mean Square Error 
        print("  MAE: %s" % mae)    # Mean Absolute Error
        print("  R2: %s" % r2)      # Correlation Measure

        mlflow.log_param("alpha", self.alpha)
        mlflow.log_param("l1_ratio", self.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model", registered_model_name=self.MODEL_NAME, signature=model_signature)
