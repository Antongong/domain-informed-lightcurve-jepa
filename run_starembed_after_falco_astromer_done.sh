#!/usr/bin/env bash
set -euo pipefail

FALCO_CKPT=${FALCO_CKPT:-/home/rui/code/algorithm_base/timeseries/falco/runs_falco/falco_leaves_no_starembed_overlap_exp10_size/checkpoints/ckpt_final_step0010000.pt}
ASTROMER_CKPT=${ASTROMER_CKPT:-/home/rui/code/algorithm_base/timeseries/astromer-2/runs_astromer2/astromer2_leaves_no_starembed_overlap_exp10_size/checkpoints/ckpt_final_step0010000.pt}
WAIT_SECONDS=${WAIT_SECONDS:-300}

while [[ ! -f "$FALCO_CKPT" || ! -f "$ASTROMER_CKPT" ]]; do
  echo "[Wait] missing checkpoint(s):"
  [[ -f "$FALCO_CKPT" ]] || echo "  $FALCO_CKPT"
  [[ -f "$ASTROMER_CKPT" ]] || echo "  $ASTROMER_CKPT"
  sleep "$WAIT_SECONDS"
done

exec /home/rui/code/algorithm_base/timeseries/clip_experiments/run_starembed_no_overlap_models.sh
