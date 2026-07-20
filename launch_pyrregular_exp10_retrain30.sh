#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rui/code/algorithm_base/timeseries/clip_experiments
PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$ROOT"

exec "$PYTHON" run_pyrregular_exp10_retrain_suite.py \
  --config runs/EXP10_lejepa_only_no_group/config_used.yaml \
  --ckpt runs/EXP10_lejepa_only_no_group/ckpt_final.pt \
  --out_root runs/pyrregular_uneven_suite_exp10_retrain30 \
  --epochs 30 \
  --num_gpus 8 \
  --device cuda:0 \
  --train_batch_size 0 \
  --train_batch_size_candidates 4096,3072,2048,1536,1024,768,512,384,256,192,128,64,32,16,8 \
  --classifiers logistic,knn \
  "$@"
