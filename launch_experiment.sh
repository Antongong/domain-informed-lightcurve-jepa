#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/rui/miniconda3/envs/timeseries/bin/python
ROOT=/home/rui/code/algorithm_base/timeseries/clip_experiments
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

CONFIGS=(
  exp10_lejepa_only_no_group.yaml
  exp11_no_crope_no_group.yaml
  exp12_no_erroraware_numeric_embedding_no_group.yaml
  exp13_no_gls_branch_no_group.yaml
  exp14_no_periodogram_numeric_branch_no_group.yaml
)

for config in "${CONFIGS[@]}"; do
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT/train_ddp_numeric.py" \
    --config "$ROOT/configs/$config"
done
