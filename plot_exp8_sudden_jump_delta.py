#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IN_CSV = (
    SCRIPT_DIR
    / "runs/EXP8_no_group_branch/starembed_sudden_jump_delta_lejepa_pred_term"
    / "exp8_sudden_jump_delta_lejepa_pred_term_scores.csv"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch/starembed_sudden_jump_delta_lejepa_pred_term"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EXP8 sudden-jump LeJEPA prediction-term delta from a saved per-sample CSV."
    )
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bins", type=int, default=90)
    parser.add_argument("--fig-width", type=float, default=10.8)
    parser.add_argument("--fig-height", type=float, default=6.6)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--xlim-delta", nargs=2, type=float, default=None)
    parser.add_argument("--xlim-score", nargs=2, type=float, default=None)
    return parser.parse_args()


def load_columns(path: Path) -> Dict[str, np.ndarray]:
    numeric_cols = [
        "original_lejepa_pred_term",
        "sudden_jump_lejepa_pred_term",
        "delta_lejepa_pred_term",
        "delta_lejepa_pred_term_g",
        "delta_lejepa_pred_term_r",
        "jump_delta_mag_abs",
    ]
    data: Dict[str, List[float]] = {col: [] for col in numeric_cols}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for col in numeric_cols:
                data[col].append(float(row[col]))
    return {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}


def save_delta_hist(cols: Dict[str, np.ndarray], args: argparse.Namespace) -> None:
    delta = cols["delta_lejepa_pred_term"]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), constrained_layout=True)
    ax.hist(
        delta,
        bins=int(args.bins),
        density=bool(args.density),
        color="#F58518",
        alpha=0.72,
        edgecolor="#9A4D00",
        linewidth=0.7,
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.4)
    ax.axvline(float(np.median(delta)), color="#B00020", linestyle="-", linewidth=1.4)
    ax.set_title("EXP8 Sudden Jump Delta in LeJEPA Prediction Term")
    ax.set_xlabel("Delta LeJEPA prediction term (sudden jump - original)")
    ax.set_ylabel("Density" if args.density else "Number of samples")
    if args.xlim_delta is not None:
        ax.set_xlim(args.xlim_delta)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    ax.text(
        0.98,
        0.96,
        f"n={delta.size}\nmean={np.mean(delta):.4f}\nmedian={np.median(delta):.4f}\nstd={np.std(delta):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3.0},
    )
    fig.savefig(args.out_dir / "exp8_sudden_jump_delta_lejepa_pred_term_hist.png", dpi=int(args.dpi))
    fig.savefig(args.out_dir / "exp8_sudden_jump_delta_lejepa_pred_term_hist.pdf")
    plt.close(fig)


def save_overlay_hist(cols: Dict[str, np.ndarray], args: argparse.Namespace) -> None:
    original = cols["original_lejepa_pred_term"]
    sudden = cols["sudden_jump_lejepa_pred_term"]
    edges = np.histogram_bin_edges(np.concatenate([original, sudden]), bins=int(args.bins))

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), constrained_layout=True)
    ax.hist(
        original,
        bins=edges,
        density=bool(args.density),
        label=f"Original test (n={original.size})",
        color="#4C78A8",
        alpha=0.42,
        histtype="stepfilled",
        edgecolor="#2C4D73",
        linewidth=1.2,
    )
    ax.hist(
        sudden,
        bins=edges,
        density=bool(args.density),
        label=f"Injected sudden jump (n={sudden.size})",
        color="#F58518",
        alpha=0.42,
        histtype="stepfilled",
        edgecolor="#9A4D00",
        linewidth=1.2,
    )
    ax.set_title("EXP8 LeJEPA Prediction Term: Original vs Sudden Jump")
    ax.set_xlabel("Per-target LeJEPA prediction term")
    ax.set_ylabel("Density" if args.density else "Number of samples")
    if args.xlim_score is not None:
        ax.set_xlim(args.xlim_score)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    fig.savefig(args.out_dir / "exp8_original_vs_sudden_jump_lejepa_pred_term_hist.png", dpi=int(args.dpi))
    fig.savefig(args.out_dir / "exp8_original_vs_sudden_jump_lejepa_pred_term_hist.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = load_columns(args.in_csv)
    save_delta_hist(cols, args)
    save_overlay_hist(cols, args)
    print(f"[OK] wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
