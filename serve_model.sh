#!/bin/bash

FTP_DIR="$HOME/ftp/mlflow_data/"

if [ "$#" -lt 1 ]; then
  echo "Please enter run ID"
  exit 1
fi

mlflow models serve -m $FTP_DIR/0/$1/artifacts/model -p 1234