#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/rui/miniconda3/envs/timeseries/bin/python}
ROOT=${ROOT:-/home/rui/code/algorithm_base/timeseries/clip_experiments}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRACT_NUM_GPUS=${EXTRACT_NUM_GPUS:-8}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-256}
BENCHMARKS=${BENCHMARKS:-clustering,logistic_knn,rf,bootstrap_linear,mlp,ood}

CONFIG="$ROOT/configs/exp21_lejepa_mean_pooling_no_group.yaml"
RUN_NAME="EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap"
RUN_DIR="$ROOT/runs/$RUN_NAME"
FEATURES_DIR="$ROOT/runs/${RUN_NAME}_starembed_features"
LOG_DIR="$ROOT/logs"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}.log"
PIPELINE_LOG="$LOG_DIR/${RUN_NAME}_pretrain_starembed_pipeline.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

{
  echo "[Start] $(date '+%Y-%m-%d %H:%M:%S') $RUN_NAME"
  echo "[Pretrain] config=$CONFIG run_dir=$RUN_DIR"
} > "$PIPELINE_LOG"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/train_ddp_numeric.py" \
  --config "$CONFIG" \
  --output_dir "$RUN_DIR" > "$TRAIN_LOG" 2>&1

{
  echo "[Pretrain done] $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[Extract] $RUN_NAME"
} >> "$PIPELINE_LOG"

"$PYTHON" "$ROOT/extract_starembed_embeddings.py" \
  --config "$RUN_DIR/config_used.yaml" \
  --ckpt "$RUN_DIR/ckpt_final.pt" \
  --out_dir "$FEATURES_DIR" \
  --out_info embeddings \
  --repr_mode concat \
  --batch_size "$EXTRACT_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --num_gpus "$EXTRACT_NUM_GPUS" \
  --save_y_str >> "$PIPELINE_LOG" 2>&1

{
  echo "[Extract done] $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[Benchmark] $FEATURES_DIR"
} >> "$PIPELINE_LOG"

"$PYTHON" "$ROOT/run_starembed_benchmarks.py" \
  --features_dir "$FEATURES_DIR" \
  --out_dir "$FEATURES_DIR/benchmark" \
  --feature_keys x \
  --benchmarks "$BENCHMARKS" \
  --seed 42 \
  --seeds 42 \
  --mlp_accelerator auto \
  --mlp_devices 1 \
  --mlp_num_workers "$NUM_WORKERS" >> "$PIPELINE_LOG" 2>&1

echo "[Done] $(date '+%Y-%m-%d %H:%M:%S') $RUN_NAME" >> "$PIPELINE_LOG"
