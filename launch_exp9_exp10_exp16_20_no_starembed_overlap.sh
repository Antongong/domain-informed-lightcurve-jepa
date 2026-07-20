#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

CONFIGS=(
  exp9_clip_only_no_group.yaml
  exp10_lejepa_only_no_group.yaml
  exp16_lejepa_no_crope_no_group.yaml
  exp17_lejepa_no_erroraware_numeric_embedding_no_group.yaml
  exp18_lejepa_no_raw_branch_no_group.yaml
  exp19_lejepa_no_phase_folding_branch_no_group.yaml
  exp20_lejepa_no_gls_branch_no_group.yaml
  exp21_lejepa_mean_pooling_no_group.yaml
)

RUN_NAMES=(
  EXP9_clip_only_no_group_no_starembed_overlap
  EXP10_lejepa_only_no_group_no_starembed_overlap
  EXP16_lejepa_no_crope_no_group_no_starembed_overlap
  EXP17_lejepa_no_erroraware_numeric_embedding_no_group_no_starembed_overlap
  EXP18_lejepa_no_raw_branch_no_group_no_starembed_overlap
  EXP19_lejepa_no_phase_folding_branch_no_group_no_starembed_overlap
  EXP20_lejepa_no_gls_branch_no_group_no_starembed_overlap
  EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap
)

mkdir -p "$ROOT/logs"

for idx in "${!CONFIGS[@]}"; do
  config="$ROOT/configs/${CONFIGS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  run_dir="$ROOT/runs/$run_name"
  log_path="$ROOT/logs/${run_name}.log"

  echo "[Launch] $run_name"
  echo "[Launch] $run_name" > "$log_path"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT/train_ddp_numeric.py" \
    --config "$config" \
    --output_dir "$run_dir" >> "$log_path" 2>&1
done
