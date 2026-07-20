#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
TIMESERIES_ROOT=${TIMESERIES_ROOT:-/home/rui/code/algorithm_base/timeseries}

MODEL_LABEL=${MODEL_LABEL:-EXP21}
RUN_NAME=${RUN_NAME:-EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap}
RUN_DIR=${RUN_DIR:-$ROOT/runs/$RUN_NAME}
ORIGINAL_FEATURES_DIR=${ORIGINAL_FEATURES_DIR:-$ROOT/runs/${RUN_NAME}_starembed_features}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/runs/sudden_jump_binary_probe_model_sweep}

NUM_WORKERS=${NUM_WORKERS:-8}
EXTRACT_NUM_GPUS=${EXTRACT_NUM_GPUS:-8}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-512}
FEATURE_KEY=${FEATURE_KEY:-x}
SEED=${SEED:-42}
MAX_ITER=${MAX_ITER:-2000}
C_GRID=${C_GRID:-0.01,0.03,0.1,0.3,1,3,10}
SKIP_EXISTING=${SKIP_EXISTING:-1}
ONLY_SUMMARIZE=${ONLY_SUMMARIZE:-0}

CONFIG=${CONFIG:-$RUN_DIR/config_used.yaml}
CKPT=${CKPT:-$RUN_DIR/ckpt_final.pt}

AMPS=(
  amp_0p05_0p1
  amp_0p1_0p3
  amp_0p3_0p5
  amp_0p5_1p0
)

DATA_ROOTS=(
  "$TIMESERIES_ROOT/starembed_preprocessed_injected_anomalies/sudden_jump"
  "$TIMESERIES_ROOT/starembed_preprocessed_injected_sudden_jump_amp_0p1_0p3/sudden_jump"
  "$TIMESERIES_ROOT/starembed_preprocessed_injected_sudden_jump_amp_0p3_0p5/sudden_jump"
  "$TIMESERIES_ROOT/starembed_preprocessed_injected_sudden_jump_amp_0p5_1p0/sudden_jump"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[Error] Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[Error] Missing directory: $path" >&2
    exit 1
  fi
}

summarize() {
  "$PYTHON" "$ROOT/summarize_sudden_jump_model_sweep_latex.py" \
    --result-root "$RESULT_ROOT" \
    --models "$MODEL_LABEL" \
    --f1-out-name "sudden_jump_model_sweep_${MODEL_LABEL}_test_f1_table.tex" \
    --c-out-name "sudden_jump_model_sweep_${MODEL_LABEL}_best_c_table.tex" \
    --csv-out-name "sudden_jump_model_sweep_${MODEL_LABEL}_summary.csv"

  "$PYTHON" "$ROOT/summarize_sudden_jump_model_sweep_latex.py" \
    --result-root "$RESULT_ROOT" \
    --models EXP10 EXP18 EXP19 EXP20 "$MODEL_LABEL" \
    --f1-out-name "sudden_jump_model_sweep_exp10_18_19_20_${MODEL_LABEL}_test_f1_table.tex" \
    --c-out-name "sudden_jump_model_sweep_exp10_18_19_20_${MODEL_LABEL}_best_c_table.tex" \
    --csv-out-name "sudden_jump_model_sweep_exp10_18_19_20_${MODEL_LABEL}_summary.csv"
}

cd "$ROOT"
mkdir -p "$RESULT_ROOT"

require_file "$CONFIG"
require_file "$CKPT"
require_file "$ORIGINAL_FEATURES_DIR/starembed_embeddings_train.npz"
require_file "$ORIGINAL_FEATURES_DIR/starembed_embeddings_validation.npz"
require_file "$ORIGINAL_FEATURES_DIR/starembed_embeddings_test.npz"

if [[ "$ONLY_SUMMARIZE" == "1" ]]; then
  summarize
  exit 0
fi

for idx in "${!AMPS[@]}"; do
  amp="${AMPS[$idx]}"
  data_root="${DATA_ROOTS[$idx]}"
  out_dir="$RESULT_ROOT/$MODEL_LABEL/$amp"
  injected_features_dir="$out_dir/features"

  require_dir "$data_root"
  mkdir -p "$out_dir"

  if [[ "$SKIP_EXISTING" == "1" && -f "$injected_features_dir/starembed_embeddings_test.npz" ]]; then
    echo "[Skip extract] $MODEL_LABEL $amp: $injected_features_dir already exists"
  else
    echo "[Extract injected] $MODEL_LABEL $amp"
    "$PYTHON" "$ROOT/extract_starembed_embeddings.py" \
      --config "$CONFIG" \
      --ckpt "$CKPT" \
      --data_root "$data_root" \
      --splits train validation test \
      --out_dir "$injected_features_dir" \
      --out_info embeddings \
      --repr_mode concat \
      --batch_size "$EXTRACT_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --num_gpus "$EXTRACT_NUM_GPUS" \
      --save_y_str
  fi

  echo "[Binary probe] $MODEL_LABEL $amp"
  "$PYTHON" "$ROOT/train_sudden_jump_binary_linear_probe.py" \
    --original-features-dir "$ORIGINAL_FEATURES_DIR" \
    --injected-features-dir "$injected_features_dir" \
    --out-dir "$out_dir" \
    --feature-key "$FEATURE_KEY" \
    --c-grid "$C_GRID" \
    --max-iter "$MAX_ITER" \
    --seed "$SEED"
done

summarize
echo "[Done] EXP21 sudden-jump ablation sweep outputs: $RESULT_ROOT/$MODEL_LABEL"
