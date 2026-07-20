#!/usr/bin/env python3
"""Run EXP10 PyRregular dataset-specific retraining without baseline experiments."""
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
    "Ldfpa.h5",
    "Pamap2.h5",
    "Animals.h5",
    "GeolifeSupervised.h5",
    "Seabirds.h5",
    "Garment.h5",
    "Abf.h5",
    "Vehicles.h5",
    "Physionet2012.h5",
    "Physionet2019.h5",
    "Taxi.h5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EXP10-only PyRregular retraining and evaluation."
    )
    parser.add_argument("--config", default="runs/EXP10_lejepa_only_no_group/config_used.yaml")
    parser.add_argument("--ckpt", default="runs/EXP10_lejepa_only_no_group/ckpt_final.pt")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--datasets_json", default=None)
    parser.add_argument("--cache_dir", default=str(Path.home() / ".cache" / "pyrregular"))
    parser.add_argument("--out_root", default="runs/pyrregular_uneven_suite_exp10_retrain30")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument(
        "--train_batch_size_candidates",
        default="4096,3072,2048,1536,1024,768,512,384,256,192,128,64,32,16,8",
        help="Global SFT batch candidates. Defaults are 8x the original single-GPU schedule.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--classifiers", default="logistic,knn")
    parser.add_argument("--value_scaling", default="none", choices=["chronos2", "none"])
    parser.add_argument("--time_strategy", default="relative", choices=["rank", "original", "normalized", "relative"])
    parser.add_argument("--theta_of_light_curve", type=float, default=1000.0)
    parser.add_argument("--max_points_per_series", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune_mode", default="full", choices=["full", "head_only", "last_k"])
    parser.add_argument("--last_k_layers", type=int, default=2)
    parser.add_argument("--save_preprocessed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_datasets_arg(args: argparse.Namespace) -> List[str]:
    if args.datasets_json:
        payload = load_json(Path(args.datasets_json))
        if isinstance(payload, dict):
            raw = payload.get("periodic_subset", payload.get("datasets"))
            if raw is None:
                raise KeyError(f"{args.datasets_json} lacks 'periodic_subset' or 'datasets'.")
        elif isinstance(payload, list):
            raw = payload
        else:
            raise TypeError(f"Unsupported datasets_json payload type: {type(payload).__name__}")
        return resolve_requested_datasets(list(raw))
    return resolve_requested_datasets(args.datasets)


def run_command(cmd: List[str]) -> None:
    print(f"[Run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_summary(args: argparse.Namespace, datasets: List[str], retrain_root: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "experiment": "pyrregular_uneven_suite_exp10_retrain",
        "source_config": str(args.config),
        "source_ckpt": str(args.ckpt),
        "epochs": int(args.epochs),
        "num_gpus": int(args.num_gpus),
        "classifiers": str(args.classifiers),
        "train_batch_size": int(args.train_batch_size),
        "train_batch_size_candidates": [
            int(item.strip())
            for item in str(args.train_batch_size_candidates).split(",")
            if item.strip()
        ],
        "datasets": [],
        "failures": [],
    }
    for dataset in datasets:
        stem = Path(dataset).stem
        train_dir = retrain_root / stem / "train"
        eval_dir = retrain_root / stem / "eval"
        report_path = eval_dir / stem / "report.json"
        ckpt_path = train_dir / "ckpt_final.pt"
        config_path = train_dir / "config_used.yaml"
        tuning_path = train_dir / "tuning_stats.json"
        if not (report_path.exists() and ckpt_path.exists() and config_path.exists()):
            summary["failures"].append(
                {
                    "dataset": dataset,
                    "missing": {
                        "report": not report_path.exists(),
                        "ckpt": not ckpt_path.exists(),
                        "config": not config_path.exists(),
                    },
                }
            )
            continue
        entry: Dict[str, Any] = {
            "dataset": dataset,
            "retrained": load_json(report_path),
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
            "ckpt": str(ckpt_path),
            "config": str(config_path),
        }
        if tuning_path.exists():
            entry["tuning"] = load_json(tuning_path)
        summary["datasets"].append(entry)
    return summary


def main() -> None:
    args = parse_args()
    if not Path(args.config).exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    datasets = resolve_datasets_arg(args)
    out_root = Path(args.out_root)
    retrain_root = out_root / "retrained"
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        stem = Path(dataset).stem
        train_dir = retrain_root / stem / "train"
        eval_dir = retrain_root / stem / "eval"
        ckpt_path = train_dir / "ckpt_final.pt"
        eval_summary = eval_dir / "summary.json"

        if args.force or not ckpt_path.exists():
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
                args.value_scaling,
                "--time_strategy",
                args.time_strategy,
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
            print(f"[Skip] retrain checkpoint exists for {dataset}: {ckpt_path}", flush=True)

        if args.force or not eval_summary.exists():
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
                args.value_scaling,
                "--time_strategy",
                args.time_strategy,
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
            print(f"[Skip] eval summary exists for {dataset}: {eval_summary}", flush=True)

        summary = build_summary(args, datasets, retrain_root)
        (out_root / "retrain_only_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    summary = build_summary(args, datasets, retrain_root)
    summary_path = out_root / "retrain_only_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Done] retrain-only summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
