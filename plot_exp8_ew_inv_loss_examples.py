#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCORE_PATH = (
    SCRIPT_DIR
    / "runs/EXP8_no_group_branch/starembed_inv_loss_anom_detection/exp8_starembed_inv_loss_scores.npz"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch/starembed_inv_loss_anom_detection"


def load_pt(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def valid_rows(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.shape[1] >= 4:
        arr = arr[arr[:, 3] > 0.5]
    return arr


def best_period(periodogram: torch.Tensor) -> float:
    pg = periodogram.detach().cpu().float()
    idx = int(torch.argmax(pg[:, 1]).item())
    return float(pg[idx, 0].item())


def select_ew_examples(score_path: Path, n_each: int) -> List[Dict[str, Any]]:
    data = np.load(score_path, allow_pickle=True)
    ew_idx = np.where(data["test_y"] == 1)[0]
    ew_loss = data["test_loss"][ew_idx]
    order = np.argsort(ew_loss)

    selected: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ew_idx[order[:n_each]], start=1):
        selected.append(
            {
                "group": "low",
                "rank": rank,
                "index": int(idx),
                "inv_loss": float(data["test_loss"][idx]),
                "inv_loss_g": float(data["test_loss_g"][idx]),
                "inv_loss_r": float(data["test_loss_r"][idx]),
                "path": str(data["test_path"][idx]),
            }
        )
    for rank, idx in enumerate(ew_idx[order[-n_each:]][::-1], start=1):
        selected.append(
            {
                "group": "high",
                "rank": rank,
                "index": int(idx),
                "inv_loss": float(data["test_loss"][idx]),
                "inv_loss_g": float(data["test_loss_g"][idx]),
                "inv_loss_r": float(data["test_loss_r"][idx]),
                "path": str(data["test_path"][idx]),
            }
        )
    return selected


def plot_raw(ax: plt.Axes, item: Dict[str, Any]) -> None:
    for band, color in (("g", "#2F6B9A"), ("r", "#B3433B")):
        lc = valid_rows(item[band]["X"]["lc"])
        ax.scatter(lc[:, 0], lc[:, 1], s=7, alpha=0.7, color=color, label=band)
    ax.invert_yaxis()
    ax.set_xlabel("time")
    ax.set_ylabel("centered mag")


def plot_gls(ax: plt.Axes, item: Dict[str, Any]) -> None:
    for band, color in (("g", "#2F6B9A"), ("r", "#B3433B")):
        pg = item[band]["X"]["periodogram"].detach().cpu().float().numpy()
        bp = best_period(item[band]["X"]["periodogram"])
        ax.plot(pg[:, 0], pg[:, 1], linewidth=1.0, alpha=0.8, color=color, label=f"{band} P={bp:.4g}")
        ax.axvline(bp, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("log10 GLS power")


def plot_phase(ax: plt.Axes, item: Dict[str, Any]) -> None:
    for band, color in (("g", "#2F6B9A"), ("r", "#B3433B")):
        pf = valid_rows(item[band]["X"]["phase_folded_lc"])
        bp = max(best_period(item[band]["X"]["periodogram"]), 1.0e-12)
        phase = np.mod(pf[:, 0] / bp, 1.0)
        ax.scatter(phase, pf[:, 1], s=7, alpha=0.68, color=color, label=band)
        ax.scatter(phase + 1.0, pf[:, 1], s=7, alpha=0.35, color=color)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("phase")
    ax.set_ylabel("centered mag")


def plot_examples(examples: List[Dict[str, Any]], out_png: Path, out_pdf: Path) -> None:
    nrows = len(examples)
    fig, axes = plt.subplots(nrows, 3, figsize=(14.5, 2.35 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])

    for row, example in enumerate(examples):
        item = load_pt(example["path"])
        source_id = str(item.get("meta", {}).get("sourceid", Path(example["path"]).stem))

        plot_raw(axes[row, 0], item)
        plot_gls(axes[row, 1], item)
        plot_phase(axes[row, 2], item)

        side = "LOW" if example["group"] == "low" else "HIGH"
        axes[row, 0].set_title(
            f"{side} #{example['rank']} idx={example['index']} loss={example['inv_loss']:.3f}\nsource={source_id}",
            fontsize=10,
        )
        axes[row, 1].set_title("GLS periodogram", fontsize=10)
        axes[row, 2].set_title("Phase-folded", fontsize=10)

        for ax in axes[row]:
            ax.grid(color="#DDDDDD", linewidth=0.7, alpha=0.75)
            ax.legend(loc="best", frameon=False, fontsize=8)

    fig.suptitle("EXP8 EW Examples by JEPA Inverse Loss", fontsize=16)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def write_selection(path: Path, examples: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "rank", "index", "inv_loss", "inv_loss_g", "inv_loss_r", "path"],
        )
        writer.writeheader()
        writer.writerows(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EW high/low inverse-loss raw, GLS, and phase-folded examples.")
    parser.add_argument("--score_path", type=Path, default=DEFAULT_SCORE_PATH)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n_each", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples = select_ew_examples(args.score_path, n_each=int(args.n_each))
    stem = f"exp8_EW_inv_loss_low_high_{int(args.n_each)}each"
    write_selection(args.out_dir / f"{stem}_selection.csv", examples)
    plot_examples(
        examples,
        out_png=args.out_dir / f"{stem}_raw_gls_phase.png",
        out_pdf=args.out_dir / f"{stem}_raw_gls_phase.pdf",
    )
    for example in examples:
        print(
            f"{example['group']} #{example['rank']}: idx={example['index']} "
            f"loss={example['inv_loss']:.6f} path={example['path']}"
        )


if __name__ == "__main__":
    main()
