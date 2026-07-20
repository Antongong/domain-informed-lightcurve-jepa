#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
CLIP_ROOT=${CLIP_ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
FALCO_ROOT=${FALCO_ROOT:-/home/rui/code/algorithm_base/timeseries/falco}
ASTROMER_ROOT=${ASTROMER_ROOT:-/home/rui/code/algorithm_base/timeseries/astromer-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
EXTRACT_NUM_GPUS=${EXTRACT_NUM_GPUS:-8}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}

mkdir -p "$CLIP_ROOT/logs" "$FALCO_ROOT/logs" "$ASTROMER_ROOT/logs"

echo "[Train] FALCO no-overlap exp10-size"
(
  cd "$FALCO_ROOT"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$FALCO_ROOT/train_falco_ddp.py" \
    --config "$FALCO_ROOT/falco_leaves_no_starembed_overlap_exp10_size.yaml"
) 2>&1 | tee "$FALCO_ROOT/logs/falco_leaves_no_starembed_overlap_exp10_size.log"

echo "[Train] Astromer-2 no-overlap exp10-size"
(
  cd "$ASTROMER_ROOT"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ASTROMER_ROOT/train_astromer2_ddp.py" \
    --config "$ASTROMER_ROOT/conf/astromer2_leaves_no_starembed_overlap_exp10_size.yaml"
) 2>&1 | tee "$ASTROMER_ROOT/logs/astromer2_leaves_no_starembed_overlap_exp10_size.log"

echo "[StarEmbed] Extract + benchmark no-overlap models"
"$CLIP_ROOT/run_starembed_no_overlap_models.sh"
