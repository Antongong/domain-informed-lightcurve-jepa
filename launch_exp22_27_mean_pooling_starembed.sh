#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRACT_NUM_GPUS=${EXTRACT_NUM_GPUS:-8}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-256}
BENCHMARKS=${BENCHMARKS:-clustering,logistic_knn,rf,bootstrap_linear,mlp,ood}
SKIP_EXISTING=${SKIP_EXISTING:-1}

CONFIGS=(
  exp22_lejepa_no_crope_mean_pooling_no_group.yaml
  exp23_lejepa_no_erroraware_numeric_embedding_mean_pooling_no_group.yaml
  exp24_lejepa_no_raw_branch_mean_pooling_no_group.yaml
  exp25_lejepa_no_phase_folding_branch_mean_pooling_no_group.yaml
  exp26_lejepa_no_gls_branch_mean_pooling_no_group.yaml
  exp27_clip_only_mean_pooling_no_group.yaml
)

RUN_NAMES=(
  EXP22_lejepa_no_crope_mean_pooling_no_group_no_starembed_overlap
  EXP23_lejepa_no_erroraware_numeric_embedding_mean_pooling_no_group_no_starembed_overlap
  EXP24_lejepa_no_raw_branch_mean_pooling_no_group_no_starembed_overlap
  EXP25_lejepa_no_phase_folding_branch_mean_pooling_no_group_no_starembed_overlap
  EXP26_lejepa_no_gls_branch_mean_pooling_no_group_no_starembed_overlap
  EXP27_clip_only_mean_pooling_no_group_no_starembed_overlap
)

cd "$ROOT"
mkdir -p "$ROOT/logs"

for idx in "${!RUN_NAMES[@]}"; do
  config="$ROOT/configs/${CONFIGS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  run_dir="$ROOT/runs/$run_name"
  features_dir="$ROOT/runs/${run_name}_starembed_features"
  log_path="$ROOT/logs/${run_name}_starembed.log"

  if [[ ! -f "$config" ]]; then
    echo "[Error] missing config: $config" >&2
    exit 1
  fi
  if [[ ! -f "$run_dir/ckpt_final.pt" ]]; then
    echo "[Error] missing checkpoint: $run_dir/ckpt_final.pt" >&2
    exit 1
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$features_dir/benchmark/summary.json" ]]; then
    echo "[Skip] $run_name already has benchmark summary"
    continue
  fi

  {
    echo "[Start] $(date '+%Y-%m-%d %H:%M:%S') $run_name"
    echo "[Config] $config"
    echo "[Run dir] $run_dir"
    echo "[Features] $features_dir"
    echo "[Extract] $run_name"
  } > "$log_path"

  "$PYTHON" "$ROOT/extract_starembed_embeddings.py" \
    --config "$run_dir/config_used.yaml" \
    --ckpt "$run_dir/ckpt_final.pt" \
    --out_dir "$features_dir" \
    --out_info embeddings \
    --repr_mode concat \
    --batch_size "$EXTRACT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_gpus "$EXTRACT_NUM_GPUS" \
    --save_y_str >> "$log_path" 2>&1

  {
    echo "[Extract done] $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[Benchmark] $features_dir"
  } >> "$log_path"

  "$PYTHON" "$ROOT/run_starembed_benchmarks.py" \
    --features_dir "$features_dir" \
    --out_dir "$features_dir/benchmark" \
    --feature_keys x \
    --benchmarks "$BENCHMARKS" \
    --seed 42 \
    --seeds 42 \
    --mlp_accelerator auto \
    --mlp_devices 1 \
    --mlp_num_workers "$NUM_WORKERS" >> "$log_path" 2>&1

  echo "[Done] $(date '+%Y-%m-%d %H:%M:%S') $run_name" >> "$log_path"
done
