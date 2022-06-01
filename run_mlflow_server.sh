#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 FTP_USERNAME FTP_PASSWORD"
    exit 1
fi

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ftp://$1:$2@localhost/mlflow_data  --host 127.0.0.1 --port 5000
