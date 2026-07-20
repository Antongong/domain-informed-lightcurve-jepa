#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rui/code/algorithm_base/timeseries/clip_experiments
PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$ROOT"

exec "$PYTHON" run_pyrregular_exp10_retrain_suite.py \
  --config runs/EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap/config_used.yaml \
  --ckpt runs/EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap/ckpt_final.pt \
  --out_root runs/pyrregular_uneven_suite_exp21_match_exp8 \
  --epochs 30 \
  --num_gpus 1 \
  --device cuda:0 \
  --train_batch_size 0 \
  --train_batch_size_candidates 512,384,256,192,128,96,64,48,32,24,16,8,4,2,1 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --classifiers logistic,mlp,knn \
  "$@"
