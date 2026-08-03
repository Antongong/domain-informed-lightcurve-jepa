#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


FEATURE_TO_SCENARIO = {
    "x": "concat",
}


def detect_feature_keys(features_dir: Path) -> List[str]:
    train_path = features_dir / "starembed_embeddings_train.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path}. Run extract_starembed_embeddings.py first or pass the correct features directory."
        )

    with np.load(train_path, allow_pickle=True) as data:
        keys = [key for key in FEATURE_TO_SCENARIO if key in data]

    if not keys:
        raise RuntimeError(f"No supported feature keys found in {train_path}. Expected one of {sorted(FEATURE_TO_SCENARIO)}.")
    return keys


def build_commands(
    *,
    repo_root: Path,
    features_dir: Path,
    out_dir: Path,
    scenario: str,
    seed: int,
    seeds: str,
    benchmarks: List[str],
    mlp_accelerator: str,
    mlp_devices: str,
    mlp_num_workers: int,
    test_features_dir: Path = None,
) -> Dict[str, List[str]]:
    my_star_embed_dir = repo_root / "my_star_embed"
    python = sys.executable

    commands: Dict[str, List[str]] = {}
    if "clustering" in benchmarks:
        commands["clustering"] = [
            python,
            str(my_star_embed_dir / "clustering.py"),
            "--emb_dir",
            str(features_dir),
            "--scenario",
            scenario,
            "--out_dir",
            str(out_dir / "clustering"),
            "--random_state",
            str(seed),
            "--mode",
            "all",
            "--clustering-method",
            "both",
        ]

    if "logistic_knn" in benchmarks:
        commands["logistic_knn"] = [
            python,
            str(my_star_embed_dir / "logistic_knn.py"),
            "--emb_dir",
            str(features_dir),
            "--scenario",
            scenario,
            "--out_dir",
            str(out_dir / "logistic_knn"),
            "--seeds",
            str(seed),
        ]
        if test_features_dir is not None:
            commands["logistic_knn"] += ["--test_emb_dir", str(test_features_dir)]

    if "rf" in benchmarks:
        commands["rf"] = [
            python,
            str(my_star_embed_dir / "rf.py"),
            "--emb_dir",
            str(features_dir),
            "--scenario",
            scenario,
            "--out_dir",
            str(out_dir / "rf"),
            "--seed",
            str(seed),
        ]
        if test_features_dir is not None:
            commands["rf"] += ["--test_emb_dir", str(test_features_dir)]

    if "bootstrap_linear" in benchmarks:
        commands["bootstrap_linear"] = [
            python,
            str(my_star_embed_dir / "bootstrap_linear.py"),
            "--emb_dir",
            str(features_dir),
            "--scenario",
            scenario,
            "--out_dir",
            str(out_dir / "bootstrap_linear"),
            "--seed",
            str(seed),
        ]

    if "mlp" in benchmarks:
        commands["mlp"] = [
            python,
            str(my_star_embed_dir / "mlp.py"),
            "--emb_dir",
            str(features_dir),
            "--scenario",
            scenario,
            "--out_dir",
            str(out_dir / "mlp"),
            "--seed",
            str(seed),
            "--accelerator",
            mlp_accelerator,
            "--devices",
            mlp_devices,
            "--num_workers",
            str(mlp_num_workers),
        ]
        if test_features_dir is not None:
            commands["mlp"] += ["--test_emb_dir", str(test_features_dir)]

    if "ood" in benchmarks:
        commands["ood"] = [
            python,
            str(my_star_embed_dir / "OOD.py"),
            "--emb_dir",
            str(features_dir),
            "--scenario",
            scenario,
            "--seeds",
            seeds,
            "--out_dir",
            str(out_dir / "ood"),
        ]

    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the StarEmbed benchmarks one by one using the implementations under my_star_embed."
    )
    parser.add_argument("--features_dir", required=True, help="Directory produced by extract_starembed_embeddings.py")
    parser.add_argument(
        "--test_features_dir",
        default="",
        help="Optional: evaluate on test data from a different directory than --features_dir "
        "(train/validation still come from --features_dir). Only supported for "
        "logistic_knn, rf, and mlp.",
    )
    parser.add_argument(
        "--feature_keys",
        default="",
        help="Comma-separated feature keys to run one by one. Default: x only.",
    )
    parser.add_argument(
        "--benchmarks",
        default="clustering,logistic_knn,rf,bootstrap_linear,mlp,ood",
        help="Comma-separated subset of clustering,logistic_knn,rf,bootstrap_linear,mlp,ood.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Single-seed benchmarks use this seed.")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds for benchmarks that still accept multiple seeds.")
    parser.add_argument("--out_dir", default="", help="Default: <features_dir>/benchmark")
    parser.add_argument("--mlp_accelerator", default="auto")
    parser.add_argument("--mlp_devices", default="auto")
    parser.add_argument("--mlp_num_workers", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_dir = Path(args.features_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else features_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_keys = (
        [key.strip() for key in args.feature_keys.split(",") if key.strip()]
        if args.feature_keys.strip()
        else detect_feature_keys(features_dir)
    )
    unknown = [key for key in feature_keys if key not in FEATURE_TO_SCENARIO]
    if unknown:
        raise KeyError(f"Unsupported feature keys: {unknown}. Supported keys: {sorted(FEATURE_TO_SCENARIO)}")

    benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    test_features_dir = Path(args.test_features_dir).resolve() if args.test_features_dir.strip() else None
    if test_features_dir is not None:
        unsupported = sorted(set(benchmarks) & {"clustering", "bootstrap_linear", "ood"})
        if unsupported:
            raise ValueError(
                f"--test_features_dir is not supported for: {unsupported} "
                "(only logistic_knn, rf, and mlp accept --test_emb_dir). "
                "Remove these from --benchmarks or drop --test_features_dir."
            )

    manifest: Dict[str, object] = {
        "features_dir": str(features_dir),
        "test_features_dir": str(test_features_dir) if test_features_dir else None,
        "feature_keys": feature_keys,
        "benchmarks": benchmarks,
        "seed": int(args.seed),
        "seeds": args.seeds,
        "runs": [],
    }

    for feature_key in feature_keys:
        scenario = FEATURE_TO_SCENARIO[feature_key]
        scenario_out_dir = out_dir / feature_key
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        commands = build_commands(
            repo_root=repo_root,
            features_dir=features_dir,
            out_dir=scenario_out_dir,
            scenario=scenario,
            seed=int(args.seed),
            seeds=args.seeds,
            benchmarks=benchmarks,
            mlp_accelerator=args.mlp_accelerator,
            mlp_devices=args.mlp_devices,
            mlp_num_workers=int(args.mlp_num_workers),
            test_features_dir=test_features_dir,
        )

        for benchmark_name, command in commands.items():
            run_record = {
                "feature_key": feature_key,
                "scenario": scenario,
                "benchmark": benchmark_name,
                "command": command,
                "cwd": str(repo_root),
            }
            manifest["runs"].append(run_record)

            print(f"[Run] feature_key={feature_key} benchmark={benchmark_name}", flush=True)
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue

            subprocess.run(command, cwd=repo_root, check=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
