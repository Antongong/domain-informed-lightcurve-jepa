#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/rui/miniconda3/envs/timeseries/bin/python
ROOT=/home/rui/code/algorithm_base/timeseries/clip_experiments
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

RUN_NAME=EXP15_no_raw_numeric_branch_no_group
CONFIG="$ROOT/configs/exp15_no_raw_numeric_branch_no_group.yaml"
RUN_DIR="$ROOT/runs/$RUN_NAME"
FEATURES_DIR="$ROOT/runs/${RUN_NAME}_starembed_features"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/train_ddp_numeric.py" \
  --config "$CONFIG"

"$PYTHON" "$ROOT/extract_starembed_embeddings.py" \
  --config "$RUN_DIR/config_used.yaml" \
  --ckpt "$RUN_DIR/ckpt_final.pt" \
  --out_dir "$FEATURES_DIR" \
  --out_info embeddings \
  --repr_mode concat \
  --views_order periodogram phase_folded \
  --batch_size 256 \
  --num_workers 8 \
  --num_gpus 8 \
  --save_y_str

"$PYTHON" "$ROOT/run_starembed_benchmarks.py" \
  --features_dir "$FEATURES_DIR" \
  --out_dir "$FEATURES_DIR/benchmark" \
  --feature_keys x \
  --benchmarks clustering,logistic_knn,rf,bootstrap_linear,mlp,ood \
  --seed 42 \
  --seeds 42 \
  --mlp_accelerator auto \
  --mlp_devices 1 \
  --mlp_num_workers 8
