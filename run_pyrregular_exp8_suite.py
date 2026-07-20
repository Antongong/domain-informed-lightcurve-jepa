#!/usr/bin/env python3
"""
Run the PyRregular uneven-suite reproduction with the clip_experiments EXP8 setup.

Purpose
-------
This launcher reproduces the experiment located at:
  /home/rui/code/algorithm_base/timeseries/lejepa_phase_folding_chronos_hybrid/runs/pyrregular_uneven_suite
but swaps the source model/config to the local EXP8 no-group-branch setup:
  config: runs/EXP8_no_group_branch/config_used.yaml
  checkpoint: runs/EXP8_no_group_branch/ckpt_final.pt

Outputs
-------
The default output directory is:
  runs/pyrregular_uneven_suite_exp8

It contains:
  baseline/summary.json
    0-shot results from the frozen EXP8 raw encoder. Reports are also written per
    dataset under baseline/<DatasetStem>/report.json.

  retrained/<DatasetStem>/train/
    Dataset-specific self-supervised fine-tuning artifacts, including
    config_used.yaml, tuning_stats.json, train_metrics.jsonl, and ckpt_final.pt.

  retrained/<DatasetStem>/eval/summary.json
    Evaluation of the SFT checkpoint on the same dataset.

  combined_summary.json
    One JSON document joining baseline and SFT reports.

Default datasets
----------------
The requested dataset list matches the referenced uneven suite:
  Mimic3.h5, Physionet2012.h5, Physionet2019.h5, Ldfpa.h5, Pamap2.h5,
  Animals.h5, GeolifeSupervised.h5, Seabirds.h5, Taxi.h5, Vehicles.h5,
  Garment.h5, Abf.h5

0-shot protocol
---------------
The 0-shot phase calls benchmark_pyrregular.py with the EXP8 config/checkpoint:
  classifiers: logistic,mlp,knn
  eval batch size: 64
  value scaling: chronos2
  time strategy: rank
  channel pooling: mean
  device: cuda:0
  GPUs: 8
  max points per series: 1000
  seed: 42

Each multivariate PyRregular sample is encoded channel by channel through the
EXP8 raw light-curve branch only. Per-channel embeddings are mean-pooled into one
sample feature vector, then scikit-learn logistic regression, MLP, and KNN are
trained on the dataset's non-test split and evaluated on its test split.

SFT protocol
------------
For each dataset, the SFT phase calls pretrain_pyrregular.py initialized from the
EXP8 checkpoint:
  epochs: 30
  train batch size: 0, which enables OOM-safe auto-tuning over
    512,384,256,192,128,96,64,48,32,24,16,8,4,2,1
  fine-tune mode: full
  value scaling: none
  time strategy: relative
  theta_of_light_curve: 1000.0
  max points per series: 1000
  num workers: 4
  seed: 42

During SFT, training channels from the non-test split are converted to raw
light-curve tensors, dataset-specific value/time/periodogram bounds are estimated,
and EXP8 is optimized with its original tri-view loss configuration:
  views_order: raw, periodogram, phase_folded
  LeJEPA enabled, weight 1, lambda 0.02
  CLIP enabled, weight 1.0, temperature 0.2, symmetric
  CLIP pairs: raw-periodogram and raw-phase_folded
  SigReg knots 17, max_t 3.0, proj_dim 256

After SFT, benchmark_pyrregular.py evaluates the dataset-specific checkpoint with:
  classifiers: logistic,mlp,knn
  value scaling: none
  time strategy: relative
  channel pooling: mean

Resume behavior
---------------
By default, existing reports/checkpoints are reused, so the command can be
stopped and resumed. Pass --force to recompute baseline, SFT, and evaluations.

Example
-------
  python run_pyrregular_exp8_suite.py

  python run_pyrregular_exp8_suite.py \\
    --datasets Mimic3.h5 Ldfpa.h5 \\
    --out_root runs/pyrregular_uneven_suite_exp8_smoke \\
    --epochs 1 --num_gpus 1 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from pyrregular_utils import resolve_requested_datasets


DEFAULT_DATASETS = [
    "Mimic3.h5",
    "Physionet2012.h5",
    "Physionet2019.h5",
    "Ldfpa.h5",
    "Pamap2.h5",
    "Animals.h5",
    "GeolifeSupervised.h5",
    "Seabirds.h5",
    "Taxi.h5",
    "Vehicles.h5",
    "Garment.h5",
    "Abf.h5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 0-shot and SFT PyRregular uneven-suite experiments with the EXP8 checkpoint."
    )
    parser.add_argument("--config", default="runs/EXP8_no_group_branch/config_used.yaml")
    parser.add_argument("--ckpt", default="runs/EXP8_no_group_branch/ckpt_final.pt")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--datasets_json",
        default=None,
        help="Optional JSON dataset list. If object, reads 'periodic_subset' first, then 'datasets'.",
    )
    parser.add_argument("--cache_dir", default=str(Path.home() / ".cache" / "pyrregular"))
    parser.add_argument("--out_root", default="runs/pyrregular_uneven_suite_exp8")
    parser.add_argument("--classifiers", default="logistic,mlp,knn")
    parser.add_argument("--baseline_value_scaling", default="chronos2", choices=["chronos2", "none"])
    parser.add_argument("--baseline_time_strategy", default="rank", choices=["rank", "original", "normalized", "relative"])
    parser.add_argument("--retrain_value_scaling", default="none", choices=["chronos2", "none"])
    parser.add_argument("--retrain_time_strategy", default="relative", choices=["rank", "original", "normalized", "relative"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--train_batch_size_candidates", default="512,384,256,192,128,96,64,48,32,24,16,8,4,2,1")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--theta_of_light_curve", type=float, default=1000.0)
    parser.add_argument("--max_points_per_series", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune_mode", default="full", choices=["full", "head_only", "last_k"])
    parser.add_argument("--last_k_layers", type=int, default=2)
    parser.add_argument("--retrain_only", action="store_true")
    parser.add_argument("--save_preprocessed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_command(cmd: List[str]) -> None:
    print(f"[Run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_datasets_arg(args: argparse.Namespace) -> List[str]:
    if args.datasets_json:
        payload = load_json(Path(args.datasets_json))
        if isinstance(payload, dict):
            raw = payload.get("periodic_subset")
            if raw is None:
                raw = payload.get("datasets")
            if raw is None:
                raise KeyError(f"{args.datasets_json} does not contain 'periodic_subset' or 'datasets'.")
        elif isinstance(payload, list):
            raw = payload
        else:
            raise TypeError(f"Unsupported datasets_json payload type: {type(payload).__name__}")
        return resolve_requested_datasets(list(raw))
    return resolve_requested_datasets(args.datasets)


def maybe_run_baseline(args: argparse.Namespace, datasets: List[str], baseline_dir: Path) -> Path:
    summary_path = baseline_dir / "summary.json"
    if args.retrain_only:
        if summary_path.exists():
            print(f"[Skip] retrain_only: reusing baseline summary at {summary_path}", flush=True)
            return summary_path
        raise FileNotFoundError(f"retrain_only requested but baseline summary is missing: {summary_path}")
    if summary_path.exists() and not args.force:
        print(f"[Skip] baseline summary already exists: {summary_path}", flush=True)
        return summary_path

    cmd = [
        sys.executable,
        "benchmark_pyrregular.py",
        "--config",
        args.config,
        "--ckpt",
        args.ckpt,
        "--datasets",
        *datasets,
        "--cache_dir",
        args.cache_dir,
        "--out_dir",
        str(baseline_dir),
        "--batch_size",
        str(args.eval_batch_size),
        "--device",
        args.device,
        "--num_gpus",
        str(args.num_gpus),
        "--classifiers",
        args.classifiers,
        "--value_scaling",
        args.baseline_value_scaling,
        "--time_strategy",
        args.baseline_time_strategy,
        "--channel_pool",
        "mean",
        "--max_points_per_series",
        str(args.max_points_per_series),
        "--seed",
        str(args.seed),
    ]
    if args.force:
        cmd.append("--force")
    run_command(cmd)
    return summary_path


def build_combined_summary(
    datasets: List[str],
    baseline_summary: Dict[str, Any],
    retrain_root: Path,
) -> Dict[str, Any]:
    baseline_map = {entry["dataset"]: entry for entry in baseline_summary.get("datasets", [])}
    combined: Dict[str, Any] = {
        "experiment": "pyrregular_uneven_suite_exp8",
        "baseline_summary": baseline_summary,
        "datasets": [],
        "failures": [],
    }

    for dataset in datasets:
        dataset_stem = Path(dataset).stem
        train_dir = retrain_root / dataset_stem / "train"
        eval_dir = retrain_root / dataset_stem / "eval"
        eval_report = eval_dir / dataset_stem / "report.json"
        train_ckpt = train_dir / "ckpt_final.pt"
        train_cfg = train_dir / "config_used.yaml"

        if not train_ckpt.exists() or not train_cfg.exists() or not eval_report.exists():
            combined["failures"].append(
                {
                    "dataset": dataset,
                    "missing": {
                        "ckpt": not train_ckpt.exists(),
                        "config": not train_cfg.exists(),
                        "eval_report": not eval_report.exists(),
                    },
                }
            )
            continue

        entry: Dict[str, Any] = {
            "dataset": dataset,
            "baseline": baseline_map.get(dataset),
            "retrained": load_json(eval_report),
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
            "ckpt": str(train_ckpt),
            "config": str(train_cfg),
        }
        tuning_stats_path = train_dir / "tuning_stats.json"
        if tuning_stats_path.exists():
            entry["tuning"] = load_json(tuning_stats_path)
        combined["datasets"].append(entry)
    return combined


def main() -> None:
    args = parse_args()
    if not Path(args.config).exists():
        raise FileNotFoundError(f"EXP8 config not found: {args.config}")
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"EXP8 checkpoint not found: {args.ckpt}")

    datasets = resolve_datasets_arg(args)
    out_root = Path(args.out_root)
    baseline_dir = out_root / "baseline"
    retrain_root = out_root / "retrained"
    combined_summary_path = out_root / "combined_summary.json"
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_summary_path = maybe_run_baseline(args, datasets, baseline_dir)
    baseline_summary = load_json(baseline_summary_path)

    for dataset in datasets:
        dataset_root = retrain_root / Path(dataset).stem
        train_dir = dataset_root / "train"
        eval_dir = dataset_root / "eval"
        eval_summary = eval_dir / "summary.json"

        if not (train_dir / "ckpt_final.pt").exists() or args.force:
            cmd = [
                sys.executable,
                "pretrain_pyrregular.py",
                "--config",
                args.config,
                "--init_ckpt",
                args.ckpt,
                "--dataset",
                dataset,
                "--cache_dir",
                args.cache_dir,
                "--out_dir",
                str(train_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.train_batch_size),
                "--batch_size_candidates",
                args.train_batch_size_candidates,
                "--num_workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--num_gpus",
                str(args.num_gpus),
                "--seed",
                str(args.seed),
                "--value_scaling",
                args.retrain_value_scaling,
                "--time_strategy",
                args.retrain_time_strategy,
                "--theta_of_light_curve",
                str(args.theta_of_light_curve),
                "--max_points_per_series",
                str(args.max_points_per_series),
                "--finetune_mode",
                args.finetune_mode,
                "--last_k_layers",
                str(args.last_k_layers),
            ]
            if args.save_preprocessed:
                cmd.append("--save_preprocessed")
            run_command(cmd)
        else:
            print(f"[Skip] retrain checkpoint already exists for {dataset}: {train_dir / 'ckpt_final.pt'}", flush=True)

        if not eval_summary.exists() or args.force:
            cmd = [
                sys.executable,
                "benchmark_pyrregular.py",
                "--config",
                str(train_dir / "config_used.yaml"),
                "--ckpt",
                str(train_dir / "ckpt_final.pt"),
                "--datasets",
                dataset,
                "--cache_dir",
                args.cache_dir,
                "--out_dir",
                str(eval_dir),
                "--batch_size",
                str(args.eval_batch_size),
                "--device",
                args.device,
                "--num_gpus",
                str(args.num_gpus),
                "--classifiers",
                args.classifiers,
                "--value_scaling",
                args.retrain_value_scaling,
                "--time_strategy",
                args.retrain_time_strategy,
                "--channel_pool",
                "mean",
                "--max_points_per_series",
                str(args.max_points_per_series),
                "--seed",
                str(args.seed),
            ]
            if args.force:
                cmd.append("--force")
            run_command(cmd)
        else:
            print(f"[Skip] eval summary already exists for {dataset}: {eval_summary}", flush=True)

        combined = build_combined_summary(datasets, baseline_summary, retrain_root)
        combined_summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    combined = build_combined_summary(datasets, baseline_summary, retrain_root)
    combined_summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"[Done] combined summary: {combined_summary_path}", flush=True)


if __name__ == "__main__":
    main()
