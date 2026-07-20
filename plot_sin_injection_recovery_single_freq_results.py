#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot true-vs-predicted recovery-head outputs for sin injection recovery.

Reads inference_train.csv, inference_val.csv, and inference_test.csv produced by
train_sin_injection_recovery_single_freq.py and writes a single multi-panel
figure for:
  P, A, phi
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/runs/"
    "EXP8_no_group_branch_sin_logP_A_phi_recovery"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sin injection recovery head results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-points-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--font-size", type=float, default=15.0)
    parser.add_argument("--legend-font-size", type=float, default=12.0)
    return parser.parse_args()


def read_inference(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    cols: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "id":
                continue
            try:
                cols.setdefault(key, []).append(float(value))
            except ValueError:
                pass
    return {key: np.asarray(values, dtype=np.float64) for key, values in cols.items()}


def maybe_sample(data: Dict[str, np.ndarray], max_points: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    if max_points <= 0:
        return data
    n = len(next(iter(data.values())))
    if n <= max_points:
        return data
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return {key: value[idx] for key, value in data.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    r = pearson(x, y)
    if not np.isfinite(r):
        return float("nan")
    return float(r * r)


def panel_specs() -> List[Tuple[str, str, str, bool]]:
    return [
        ("P_true", "P_pred", "P", True),
        ("A_true", "A_pred", "A", True),
        ("phi_true", "phi_pred", r"$\varphi$", False),
    ]


def finite_limits(xs: List[np.ndarray], ys: List[np.ndarray], axis_log: bool) -> Tuple[float, float]:
    values = np.concatenate([v[np.isfinite(v)] for v in xs + ys])
    if axis_log:
        values = values[values > 0.0]
    if values.size == 0:
        return (1.0e-6, 1.0)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.1 + 1.0e-6
        return lo - pad, hi + pad
    if axis_log:
        return lo * 0.8, hi * 1.25
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output = args.output
    if output is None:
        output = results_dir / "true_vs_pred_recovery.pdf"
    output = output.expanduser().resolve()

    rng = np.random.default_rng(args.seed)
    split_data: Dict[str, Dict[str, np.ndarray]] = {}
    splits = ("test",)
    for split in splits:
        path = results_dir / f"inference_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing inference CSV: {path}")
        split_data[split] = maybe_sample(read_inference(path), args.max_points_per_split, rng)

    colors = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
    with plt.rc_context(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size + 2,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size - 1,
            "ytick.labelsize": args.font_size - 1,
            "legend.fontsize": args.legend_font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.0), constrained_layout=True)

        for ax, (true_key, pred_key, label, axis_log) in zip(axes, panel_specs()):
            xs_all = [data[true_key] for data in split_data.values()]
            ys_all = [data[pred_key] for data in split_data.values()]
            lo, hi = finite_limits(xs_all, ys_all, axis_log=axis_log)

            for split, data in split_data.items():
                x = data[true_key]
                y = data[pred_key]
                ok = np.isfinite(x) & np.isfinite(y)
                if axis_log:
                    ok = ok & (x > 0.0) & (y > 0.0)
                r2 = r_squared(x[ok], y[ok])
                ax.scatter(
                    x[ok],
                    y[ok],
                    s=10,
                    alpha=0.35,
                    linewidths=0,
                    color=colors[split],
                    label=rf"{split} $R^2={r2:.3f}$",
                )

            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.75)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if axis_log:
                ax.set_xscale("log")
                ax.set_yscale("log")
            if label == "phi":
                ax.set_xlim(0.0, 2.0 * math.pi)
                ax.set_ylim(0.0, 2.0 * math.pi)
                ax.plot([0.0, 2.0 * math.pi], [0.0, 2.0 * math.pi], color="black", linewidth=1.0, alpha=0.75)
            ax.set_title(label)
            ax.set_xlabel("true")
            ax.set_ylabel("pred")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", frameon=False)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi)
        plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
