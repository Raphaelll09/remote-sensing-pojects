#!/bin/bash

if [ $# -eq 0 ];
  then
  export HUGGINGFACE_TOKEN=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
elif [ $# -gt 0 ];
  then
  export HUGGINGFACE_TOKEN=$1
else
  echo "Usage: source $0 <token>  OR source $0"
  exit 1
fi

conda create -n project_teledetection_05 python=3.11
conda activate project_teledetection_05
pip install -e .
python src/data_loading.py