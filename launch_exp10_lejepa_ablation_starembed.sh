#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
EXTRACT_NUM_GPUS=${EXTRACT_NUM_GPUS:-8}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}

CONFIGS=(
  exp16_lejepa_no_crope_no_group.yaml
  exp17_lejepa_no_erroraware_numeric_embedding_no_group.yaml
  exp18_lejepa_no_raw_branch_no_group.yaml
  exp19_lejepa_no_phase_folding_branch_no_group.yaml
  exp20_lejepa_no_gls_branch_no_group.yaml
  exp21_lejepa_mean_pooling_no_group.yaml
)

RUN_NAMES=(
  EXP16_lejepa_no_crope_no_group
  EXP17_lejepa_no_erroraware_numeric_embedding_no_group
  EXP18_lejepa_no_raw_branch_no_group
  EXP19_lejepa_no_phase_folding_branch_no_group
  EXP20_lejepa_no_gls_branch_no_group
  EXP21_lejepa_mean_pooling_no_group
)

for idx in "${!CONFIGS[@]}"; do
  config="$ROOT/configs/${CONFIGS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  run_dir="$ROOT/runs/$run_name"
  features_dir="$ROOT/runs/${run_name}_starembed_features"

  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT/train_ddp_numeric.py" \
    --config "$config"

  "$PYTHON" "$ROOT/extract_starembed_embeddings.py" \
    --config "$run_dir/config_used.yaml" \
    --ckpt "$run_dir/ckpt_final.pt" \
    --out_dir "$features_dir" \
    --out_info embeddings \
    --repr_mode concat \
    --batch_size "$EXTRACT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_gpus "$EXTRACT_NUM_GPUS" \
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
    --mlp_num_workers "$NUM_WORKERS"
done
