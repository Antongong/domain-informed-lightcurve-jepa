#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/rui/miniconda3/envs/timeseries/bin/python
ROOT=/home/rui/code/algorithm_base/timeseries/clip_experiments

RUNS=(
  EXP9_clip_only_no_group
  EXP10_lejepa_only_no_group
  EXP11_no_crope_no_group
  EXP12_no_erroraware_numeric_embedding_no_group
  EXP13_no_gls_branch_no_group
  EXP14_no_periodogram_numeric_branch_no_group
)

for run_name in "${RUNS[@]}"; do
  features_dir="$ROOT/runs/${run_name}_starembed_features"

  "$PYTHON" "$ROOT/extract_starembed_embeddings.py" \
    --config "$ROOT/runs/$run_name/config_used.yaml" \
    --ckpt "$ROOT/runs/$run_name/ckpt_final.pt" \
    --out_dir "$features_dir" \
    --out_info embeddings \
    --repr_mode concat \
    --batch_size 256 \
    --num_workers 8 \
    --num_gpus 8 \
    --save_y_str

  "$PYTHON" "$ROOT/run_starembed_benchmarks.py" \
    --features_dir "$features_dir" \
    --out_dir "$features_dir/benchmark" \
    --feature_keys x \
    --benchmarks clustering,logistic_knn,rf,bootstrap_linear,mlp,ood \
    --seed 42 \
    --seeds 42 \
    --mlp_accelerator auto \
    --mlp_devices 1 \
    --mlp_num_workers 8
done
