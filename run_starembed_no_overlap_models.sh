#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
CLIP_ROOT=${CLIP_ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
FALCO_ROOT=${FALCO_ROOT:-/home/rui/code/algorithm_base/timeseries/falco}
ASTROMER_ROOT=${ASTROMER_ROOT:-/home/rui/code/algorithm_base/timeseries/astromer-2}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRACT_NUM_GPUS=${EXTRACT_NUM_GPUS:-8}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-256}
BENCHMARKS=${BENCHMARKS:-clustering,logistic_knn,rf,bootstrap_linear,mlp,ood}

run_clip_benchmark() {
  local features_dir="$1"
  "$PYTHON" "$CLIP_ROOT/run_starembed_benchmarks.py" \
    --features_dir "$features_dir" \
    --out_dir "$features_dir/benchmark" \
    --feature_keys x \
    --benchmarks "$BENCHMARKS" \
    --seed 42 \
    --seeds 42 \
    --mlp_accelerator auto \
    --mlp_devices 1 \
    --mlp_num_workers "$NUM_WORKERS"
}

run_compat_benchmark() {
  local features_dir="$1"
  local prefix="$2"

  for split in train test validation anom; do
    local src="$features_dir/${prefix}_embeddings_${split}.npz"
    local dst="$features_dir/starembed_embeddings_${split}.npz"
    if [[ ! -f "$src" ]]; then
      echo "[Error] Missing extracted features: $src" >&2
      return 1
    fi
    ln -sfn "$(basename "$src")" "$dst"
  done

  run_clip_benchmark "$features_dir"
}

cd "$CLIP_ROOT"

CLIP_RUN_NAMES=(
  EXP9_clip_only_no_group_no_starembed_overlap
  EXP10_lejepa_only_no_group_no_starembed_overlap
  EXP16_lejepa_no_crope_no_group_no_starembed_overlap
  EXP17_lejepa_no_erroraware_numeric_embedding_no_group_no_starembed_overlap
  EXP18_lejepa_no_raw_branch_no_group_no_starembed_overlap
  EXP19_lejepa_no_phase_folding_branch_no_group_no_starembed_overlap
  EXP20_lejepa_no_gls_branch_no_group_no_starembed_overlap
  EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap
)

for run_name in "${CLIP_RUN_NAMES[@]}"; do
  run_dir="$CLIP_ROOT/runs/$run_name"
  features_dir="$CLIP_ROOT/runs/${run_name}_starembed_features"
  ckpt="$run_dir/ckpt_final.pt"
  config="$run_dir/config_used.yaml"

  if [[ ! -f "$ckpt" ]]; then
    echo "[Skip] Missing checkpoint for $run_name: $ckpt" >&2
    continue
  fi
  echo "[Extract] $run_name"
  "$PYTHON" "$CLIP_ROOT/extract_starembed_embeddings.py" \
    --config "$config" \
    --ckpt "$ckpt" \
    --out_dir "$features_dir" \
    --out_info embeddings \
    --repr_mode concat \
    --batch_size "$EXTRACT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_gpus "$EXTRACT_NUM_GPUS" \
    --save_y_str
  echo "[Benchmark] $run_name"
  run_clip_benchmark "$features_dir"
done

echo "[Extract] FALCO no-overlap exp10-size"
(
  cd "$FALCO_ROOT"
  "$PYTHON" "$FALCO_ROOT/extract_starembed_falco.py" \
    --config "$FALCO_ROOT/falco_leaves_no_starembed_overlap_exp10_size.yaml" \
    --ckpt "$FALCO_ROOT/runs_falco/falco_leaves_no_starembed_overlap_exp10_size/checkpoints/ckpt_final_step0010000.pt"
)
echo "[Benchmark] FALCO no-overlap exp10-size"
run_compat_benchmark "$FALCO_ROOT/starembed_embeddings_falco_no_starembed_overlap_exp10_size" falco

echo "[Extract] Astromer-2 no-overlap exp10-size"
(
  cd "$ASTROMER_ROOT"
  "$PYTHON" "$ASTROMER_ROOT/extract_starembed.py" \
    --config "$ASTROMER_ROOT/conf/astromer2_leaves_no_starembed_overlap_exp10_size.yaml" \
    --ckpt "$ASTROMER_ROOT/runs_astromer2/astromer2_leaves_no_starembed_overlap_exp10_size/checkpoints/ckpt_final_step0010000.pt" \
    --data_dir /home/rui/code/algorithm_base/timeseries/data_complete/data_complete \
    --seq_len 200 \
    --norm zscore \
    --out_dir "$ASTROMER_ROOT/starembed_embeddings_astromer2_no_starembed_overlap_exp10_size" \
    --out_prefix astromer2
)
echo "[Benchmark] Astromer-2 no-overlap exp10-size"
run_compat_benchmark "$ASTROMER_ROOT/starembed_embeddings_astromer2_no_starembed_overlap_exp10_size" astromer2
