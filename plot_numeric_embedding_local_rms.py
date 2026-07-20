#!/usr/bin/env python3
"""Plot local RMS summaries of numeric value-embedding weights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


REPO_ROOT = Path("/home/rui/code/algorithm_base/timeseries")
RUNS_DIR = REPO_ROOT / "clip_experiments/runs"
DEFAULT_OUT_DIR = RUNS_DIR / "numeric_embedding_weight_visualizations"


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    run_dir: Path


EXPERIMENTS = [
    ExperimentSpec("EXP8 error-aware numeric embedding", RUNS_DIR / "EXP8_no_group_branch"),
    ExperimentSpec(
        "EXP12 no error-aware numeric embedding",
        RUNS_DIR / "EXP12_no_erroraware_numeric_embedding_no_group",
    ),
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def get_nested(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_embedding(run_dir: Path, view: str) -> tuple[np.ndarray, np.ndarray, bool]:
    cfg = load_yaml(run_dir / "config_used.yaml")
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    view_to_key = {
        "raw": "raw_value_embed.embedding.weight",
        "phase": "phase_value_embed.embedding.weight",
        "periodogram": "period_value_embed.embedding.weight",
    }
    weight_key = view_to_key[view]
    if weight_key not in state:
        raise KeyError(f"{weight_key!r} not found in {run_dir / 'ckpt_final.pt'}")

    weight = state[weight_key].detach().float().cpu().numpy()

    numc = get_nested(cfg, ["model", "numeric"], {}) or {}
    if view == "periodogram":
        view_cfg = numc.get("periodogram", {}) or {}
        vmin = float(view_cfg.get("vmin", -6.0))
        vmax = float(view_cfg.get("vmax", 2.0))
        use_uncertainty = False
    else:
        raw_cfg = numc.get("raw", {}) or {}
        vmin = float(raw_cfg.get("vmin", -2.0))
        vmax = float(raw_cfg.get("vmax", 2.0))
        view_cfg = raw_cfg if view == "raw" else (numc.get("phase_folded", {}) or {})
        use_uncertainty = bool(view_cfg.get("use_uncertainty", True))
        ablation_unc = get_nested(cfg, ["model", "ablations", "use_uncertainty"], None)
        if ablation_unc is not None:
            use_uncertainty = bool(ablation_unc)

    values = np.linspace(vmin, vmax, weight.shape[0], dtype=np.float32)
    return values, weight, use_uncertainty


def local_rms_mean(values: np.ndarray, weight: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS over a sliding value-grid window, then mean over embedding dims."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > weight.shape[0]:
        raise ValueError(f"window={window} exceeds number of value bins={weight.shape[0]}")

    sq = weight.astype(np.float64) ** 2
    cumsum = np.concatenate([np.zeros((1, sq.shape[1]), dtype=np.float64), np.cumsum(sq, axis=0)], axis=0)
    win_mean_sq = (cumsum[window:] - cumsum[:-window]) / float(window)
    win_rms = np.sqrt(win_mean_sq)
    y = win_rms.mean(axis=1)
    x = np.convolve(values.astype(np.float64), np.ones(window, dtype=np.float64) / window, mode="valid")
    return x, y


def plot_rms(records: list[dict[str, Any]], out_path: Path, view: str, window: int) -> None:
    fig, axes = plt.subplots(len(records), 1, figsize=(12, 6.2), sharex=True, sharey=True)
    if len(records) == 1:
        axes = [axes]

    for ax, rec in zip(axes, records):
        ax.plot(rec["x"], rec["rms"], color=rec["color"], linewidth=2.0)
        suffix = "enabled" if rec["use_uncertainty"] else "disabled"
        ax.set_title(f"{rec['label']}; error-aware path {suffix}", fontsize=11)
        ax.set_ylabel("Mean local RMS")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)

    axes[-1].set_xlabel("True numeric value")
    fig.suptitle(f"{view.capitalize()} Numeric Embedding Local RMS, {window}-Bin Window", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for rec in records:
        for x, y in zip(rec["x"], rec["rms"]):
            rows.append(
                {
                    "experiment": rec["run_name"],
                    "numeric_value": float(x),
                    "mean_local_rms": float(y),
                    "window": int(rec["window"]),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--view", choices=["raw", "phase", "periodogram"], default="raw")
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#1f77b4", "#d62728"]
    records = []
    for spec, color in zip(EXPERIMENTS, colors):
        values, weight, use_uncertainty = load_embedding(spec.run_dir, args.view)
        x, rms = local_rms_mean(values, weight, args.window)
        records.append(
            {
                "label": spec.label,
                "run_name": spec.run_dir.name,
                "x": x,
                "rms": rms,
                "use_uncertainty": use_uncertainty,
                "window": args.window,
                "color": color,
            }
        )

    out_path = args.out_dir / f"EXP8_vs_EXP12_{args.view}_numeric_embedding_local_rms_w{args.window}.png"
    plot_rms(records, out_path, args.view, args.window)
    print(f"Wrote {out_path}")

    csv_path = args.out_dir / f"EXP8_vs_EXP12_{args.view}_numeric_embedding_local_rms_w{args.window}.csv"
    write_csv(records, csv_path)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
