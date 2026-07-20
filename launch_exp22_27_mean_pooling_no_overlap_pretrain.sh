#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
GENERATE_ONLY=${GENERATE_ONLY:-0}

CONFIGS=(
  exp22_lejepa_no_crope_mean_pooling_no_group.yaml
  exp23_lejepa_no_erroraware_numeric_embedding_mean_pooling_no_group.yaml
  exp24_lejepa_no_raw_branch_mean_pooling_no_group.yaml
  exp25_lejepa_no_phase_folding_branch_mean_pooling_no_group.yaml
  exp26_lejepa_no_gls_branch_mean_pooling_no_group.yaml
  exp27_clip_only_mean_pooling_no_group.yaml
)

SOURCE_CONFIGS=(
  exp16_lejepa_no_crope_no_group.yaml
  exp17_lejepa_no_erroraware_numeric_embedding_no_group.yaml
  exp18_lejepa_no_raw_branch_no_group.yaml
  exp19_lejepa_no_phase_folding_branch_no_group.yaml
  exp20_lejepa_no_gls_branch_no_group.yaml
  exp9_clip_only_no_group.yaml
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
mkdir -p "$ROOT/logs" "$ROOT/configs"

if [[ "${#CONFIGS[@]}" -ne "${#SOURCE_CONFIGS[@]}" || "${#CONFIGS[@]}" -ne "${#RUN_NAMES[@]}" ]]; then
  echo "[Error] CONFIGS, SOURCE_CONFIGS, and RUN_NAMES must have the same length." >&2
  exit 1
fi

for idx in "${!CONFIGS[@]}"; do
  src_config="$ROOT/configs/${SOURCE_CONFIGS[$idx]}"
  config="$ROOT/configs/${CONFIGS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  run_dir="$ROOT/runs/$run_name"
  log_path="$ROOT/logs/${run_name}.log"

  if [[ ! -f "$src_config" ]]; then
    echo "[Error] Missing source config: $src_config" >&2
    exit 1
  fi

  "$PYTHON" - "$src_config" "$config" "$run_dir" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
run_dir = Path(sys.argv[3])

with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg.setdefault("model", {}).setdefault("numeric", {}).setdefault("pooling", {})["mode"] = "mean"
cfg["training"]["output_dir"] = str(run_dir)

with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

  echo "[Config] wrote $config from $src_config with pooling.mode=mean"

  if [[ "$GENERATE_ONLY" == "1" ]]; then
    continue
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$run_dir/ckpt_final.pt" ]]; then
    echo "[Skip] $run_name already has $run_dir/ckpt_final.pt"
    continue
  fi

  echo "[Launch] $run_name"
  {
    echo "[Launch] $run_name"
    echo "[Config] $config"
    echo "[Output] $run_dir"
    date '+[Start] %Y-%m-%d %H:%M:%S'
  } > "$log_path"

  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT/train_ddp_numeric.py" \
    --config "$config" \
    --output_dir "$run_dir" >> "$log_path" 2>&1

  date '+[Done] %Y-%m-%d %H:%M:%S' >> "$log_path"
done
