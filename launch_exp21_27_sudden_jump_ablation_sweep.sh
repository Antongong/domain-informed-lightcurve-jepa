#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
SKIP_EXISTING=${SKIP_EXISTING:-1}
ONLY_SUMMARIZE=${ONLY_SUMMARIZE:-0}

MODELS=(
  "EXP21:EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap"
  "EXP22:EXP22_lejepa_no_crope_mean_pooling_no_group_no_starembed_overlap"
  "EXP23:EXP23_lejepa_no_erroraware_numeric_embedding_mean_pooling_no_group_no_starembed_overlap"
  "EXP24:EXP24_lejepa_no_raw_branch_mean_pooling_no_group_no_starembed_overlap"
  "EXP25:EXP25_lejepa_no_phase_folding_branch_mean_pooling_no_group_no_starembed_overlap"
  "EXP26:EXP26_lejepa_no_gls_branch_mean_pooling_no_group_no_starembed_overlap"
  "EXP27:EXP27_clip_only_mean_pooling_no_group_no_starembed_overlap"
)

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/runs/sudden_jump_binary_probe_model_sweep"

for spec in "${MODELS[@]}"; do
  label="${spec%%:*}"
  run_name="${spec#*:}"
  echo "[Run] $label $run_name"
  MODEL_LABEL="$label" \
  RUN_NAME="$run_name" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  ONLY_SUMMARIZE="$ONLY_SUMMARIZE" \
  PYTHON="$PYTHON" \
  "$ROOT/launch_exp21_sudden_jump_ablation_sweep.sh" \
    2>&1 | tee "$ROOT/logs/${label}_sudden_jump_sweep.log"
done

if [[ "$ONLY_SUMMARIZE" != "1" ]]; then
  "$PYTHON" "$ROOT/summarize_sudden_jump_model_sweep_latex.py" \
    --result-root "$ROOT/runs/sudden_jump_binary_probe_model_sweep" \
    --models EXP21 EXP22 EXP23 EXP24 EXP25 EXP26 EXP27 \
    --f1-out-name sudden_jump_model_sweep_exp21_27_test_f1_table.tex \
    --c-out-name sudden_jump_model_sweep_exp21_27_best_c_table.tex \
    --csv-out-name sudden_jump_model_sweep_exp21_27_summary.csv
fi

echo "[Done] EXP21-27 sudden-jump sweep"
