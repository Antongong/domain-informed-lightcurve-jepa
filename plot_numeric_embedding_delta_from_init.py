#!/usr/bin/env python3
"""Plot numeric value-embedding changes from initialization for EXP8 and EXP12."""

from __future__ import annotations

import argparse
import sys
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
SCRIPT_DIR = REPO_ROOT / "clip_experiments"
RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_OUT_DIR = RUNS_DIR / "numeric_embedding_weight_visualizations"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_ddp_numeric import build_model, set_seed  # noqa: E402


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


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def value_embedding_key(view: str) -> str:
    keys = {
        "raw": "raw_value_embed.embedding.weight",
        "phase": "phase_value_embed.embedding.weight",
        "periodogram": "period_value_embed.embedding.weight",
    }
    if view not in keys:
        raise ValueError(f"Unsupported view {view!r}; choose from {sorted(keys)}")
    return keys[view]


def numeric_value_grid(cfg: dict[str, Any], view: str, n_bins: int) -> tuple[np.ndarray, bool]:
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
    return np.linspace(vmin, vmax, n_bins, dtype=np.float32), use_uncertainty


def load_final_and_initial(spec: ExperimentSpec, view: str) -> dict[str, Any]:
    cfg = load_yaml(spec.run_dir / "config_used.yaml")
    seed = int(get_nested(cfg, ["training", "seed"], 1234))
    key = value_embedding_key(view)

    set_seed(seed)
    init_model = build_model(cfg)
    init_state = init_model.state_dict()
    if key not in init_state:
        raise KeyError(f"{key!r} not found in reconstructed initial model")
    init = init_state[key].detach().float().cpu().numpy()

    ckpt = torch.load(spec.run_dir / "ckpt_final.pt", map_location="cpu")
    final_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if key not in final_state:
        raise KeyError(f"{key!r} not found in {spec.run_dir / 'ckpt_final.pt'}")
    final = final_state[key].detach().float().cpu().numpy()

    if final.shape != init.shape:
        raise ValueError(f"Shape mismatch for {spec.run_dir.name}: final={final.shape}, init={init.shape}")

    values, use_uncertainty = numeric_value_grid(cfg, view, final.shape[0])
    delta = final - init
    return {
        "label": spec.label,
        "run_name": spec.run_dir.name,
        "values": values,
        "init": init,
        "final": final,
        "delta": delta,
        "use_uncertainty": use_uncertainty,
    }


def local_rms_mean(values: np.ndarray, weight: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    sq = weight.astype(np.float64) ** 2
    cumsum = np.concatenate([np.zeros((1, sq.shape[1]), dtype=np.float64), np.cumsum(sq, axis=0)], axis=0)
    win_mean_sq = (cumsum[window:] - cumsum[:-window]) / float(window)
    win_rms = np.sqrt(win_mean_sq)
    y = win_rms.mean(axis=1)
    x = np.convolve(values.astype(np.float64), np.ones(window, dtype=np.float64) / window, mode="valid")
    return x, y


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1.0e-12)
    return np.sum(a * b, axis=1) / denom


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")


def pca_fit_transform(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = mat.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]
    scores = centered @ components.T
    explained = singular_values**2 / max(centered.shape[0] - 1, 1)
    explained_ratio = explained[:3] / explained.sum()
    return scores, explained_ratio


def scores_to_rgb(scores: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    lo = np.percentile(scores, low_pct, axis=0)
    hi = np.percentile(scores, high_pct, axis=0)
    denom = np.maximum(hi - lo, 1.0e-12)
    return np.clip((scores - lo) / denom, 0.0, 1.0)


def plot_delta_summary(records: list[dict[str, Any]], out_path: Path, view: str) -> None:
    colors = ["#1f77b4", "#d62728"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)

    for rec, color in zip(records, colors):
        values = rec["values"]
        delta = rec["delta"]
        init = rec["init"]
        final = rec["final"]
        rms = np.sqrt(np.mean(delta.astype(np.float64) ** 2, axis=1))
        l2 = np.linalg.norm(delta, axis=1)
        one_minus_cos = 1.0 - cosine_rows(final, init)

        axes[0].plot(values, rms, label=rec["label"], color=color, linewidth=1.5)
        axes[1].plot(values, l2, label=rec["label"], color=color, linewidth=1.5)
        axes[2].plot(values, one_minus_cos, label=rec["label"], color=color, linewidth=1.5)

    axes[0].set_ylabel("RMS(Delta E)")
    axes[1].set_ylabel("||Delta E||2")
    axes[2].set_ylabel("1 - cos(E_final, E_init)")
    axes[2].set_xlabel("True numeric value")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_delta_heatmap(records: list[dict[str, Any]], out_path: Path, view: str, contrast_percentile: float) -> None:
    color_abs_max = float(np.percentile(np.abs(np.concatenate([rec["delta"].ravel() for rec in records])), contrast_percentile))
    color_abs_max = max(color_abs_max, 1.0e-9)

    fig, axes = plt.subplots(1, len(records), figsize=(14.5, 5.3), sharey=True)
    if len(records) == 1:
        axes = [axes]

    im = None
    for ax, rec in zip(axes, records):
        # Transpose so x is numeric value and y is embedding dim.
        delta_map = rec["delta"].T
        im = ax.imshow(
            delta_map,
            aspect="auto",
            origin="lower",
            extent=[float(rec["values"][0]), float(rec["values"][-1]), 0, delta_map.shape[0] - 1],
            cmap="seismic",
            vmin=-color_abs_max,
            vmax=color_abs_max,
            interpolation="nearest",
        )
        ax.set_xlabel("True numeric value")

    axes[0].set_ylabel("Embedding dim")
    assert im is not None
    cbar = fig.colorbar(im, ax=axes, pad=0.015)
    cbar.set_label(r"$\Delta$Emb")
    save_figure(fig, out_path)
    plt.close(fig)


def plot_local_rms(records: list[dict[str, Any]], out_path: Path, view: str, window: int) -> None:
    colors = ["#1f77b4", "#d62728"]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for rec, color in zip(records, colors):
        x, y = local_rms_mean(rec["values"], rec["delta"], window)
        ax.plot(x, y, color=color, linewidth=2.6, label=rec["label"])
    ax.set_ylabel(r"Mean local RMS($\Delta$Emb)")
    ax.set_xlabel("True numeric value")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_pca_rgb(
    records: list[dict[str, Any]],
    out_path: Path,
    view: str,
    low_pct: float,
    high_pct: float,
    strip_height: int,
    fig_height: float,
    left_margin: float,
) -> None:
    rgbs = []
    explained_ratios = []
    for rec in records:
        scores, explained_ratio = pca_fit_transform(rec["delta"])
        rgbs.append(scores_to_rgb(scores, low_pct, high_pct))
        explained_ratios.append(explained_ratio)

    fig, axes = plt.subplots(len(records), 1, figsize=(13, fig_height), sharex=True)
    if len(records) == 1:
        axes = [axes]

    for ax, rec, rgb in zip(axes, records, rgbs):
        strip = np.repeat(rgb[np.newaxis, :, :], strip_height, axis=0)
        ax.imshow(
            strip,
            aspect="auto",
            origin="lower",
            extent=[float(rec["values"][0]), float(rec["values"][-1]), 0.0, 1.0],
            interpolation="nearest",
        )
        ax.set_yticks([])

    axes[-1].set_xlabel("True numeric value")
    fig.supylabel(r"$\Delta$Emb Principal Components", x=0.015)
    fig.tight_layout(rect=(left_margin, 0, 1, 1))
    save_figure(fig, out_path)
    plt.close(fig)

    for rec, ratio in zip(records, explained_ratios):
        print(
            f"{rec['run_name']} independent PCA explained ratio: "
            f"PC1={ratio[0]:.6f}, PC2={ratio[1]:.6f}, PC3={ratio[2]:.6f}"
        )


def write_delta_csv(records: list[dict[str, Any]], out_path: Path, window: int) -> None:
    rows = []
    for rec in records:
        rms = np.sqrt(np.mean(rec["delta"].astype(np.float64) ** 2, axis=1))
        l2 = np.linalg.norm(rec["delta"], axis=1)
        one_minus_cos = 1.0 - cosine_rows(rec["final"], rec["init"])
        local_x, local_y = local_rms_mean(rec["values"], rec["delta"], window)
        local_map = dict(zip(np.round(local_x, 8), local_y))
        for value, r, n, c in zip(rec["values"], rms, l2, one_minus_cos):
            rows.append(
                {
                    "experiment": rec["run_name"],
                    "numeric_value": float(value),
                    "delta_rms": float(r),
                    "delta_l2": float(n),
                    "one_minus_cos_final_init": float(c),
                    "local_rms_w_centered": np.nan,
                    "window": int(window),
                }
            )
        for value, y in local_map.items():
            rows.append(
                {
                    "experiment": rec["run_name"],
                    "numeric_value": float(value),
                    "delta_rms": np.nan,
                    "delta_l2": np.nan,
                    "one_minus_cos_final_init": np.nan,
                    "local_rms_w_centered": float(y),
                    "window": int(window),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--view", choices=["raw", "phase", "periodogram"], default="raw")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--low_pct", type=float, default=1.0)
    parser.add_argument("--high_pct", type=float, default=99.0)
    parser.add_argument("--strip_height", type=int, default=72)
    parser.add_argument("--pca_fig_height", type=float, default=3.8)
    parser.add_argument("--pca_left_margin", type=float, default=0.035)
    parser.add_argument("--run_dirs", type=Path, nargs=2, default=None)
    parser.add_argument("--labels", type=str, nargs=2, default=None)
    parser.add_argument("--output_stem", type=str, default=None)
    parser.add_argument("--only_pca", action="store_true")
    parser.add_argument(
        "--contrast_percentile",
        type=float,
        default=99.0,
        help="Symmetric heatmap color limit percentile for abs(Delta E).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dirs is None:
        specs = EXPERIMENTS
    else:
        labels = args.labels or [path.name for path in args.run_dirs]
        specs = [ExperimentSpec(label, run_dir) for label, run_dir in zip(labels, args.run_dirs)]

    records = [load_final_and_initial(spec, args.view) for spec in specs]

    stem = args.output_stem or f"EXP8_vs_EXP12_{args.view}_numeric_embedding_delta_from_init"
    if not args.only_pca:
        summary_path = args.out_dir / f"{stem}_summary.png"
        plot_delta_summary(records, summary_path, args.view)
        print(f"Wrote {summary_path}")

        heatmap_path = args.out_dir / f"{stem}_weights.png"
        plot_delta_heatmap(records, heatmap_path, args.view, args.contrast_percentile)
        print(f"Wrote {heatmap_path}")

        local_rms_path = args.out_dir / f"{stem}_local_rms_w{args.window}.png"
        plot_local_rms(records, local_rms_path, args.view, args.window)
        print(f"Wrote {local_rms_path}")

    pca_path = args.out_dir / f"{stem}_pca_rgb_strip.png"
    plot_pca_rgb(
        records,
        pca_path,
        args.view,
        args.low_pct,
        args.high_pct,
        args.strip_height,
        args.pca_fig_height,
        args.pca_left_margin,
    )
    print(f"Wrote {pca_path}")

    if not args.only_pca:
        csv_path = args.out_dir / f"{stem}_metrics.csv"
        write_delta_csv(records, csv_path, args.window)
        print(f"Wrote {csv_path}")

    for rec in records:
        delta_rms = float(np.sqrt(np.mean(rec["delta"].astype(np.float64) ** 2)))
        init_rms = float(np.sqrt(np.mean(rec["init"].astype(np.float64) ** 2)))
        final_rms = float(np.sqrt(np.mean(rec["final"].astype(np.float64) ** 2)))
        print(f"{rec['run_name']}: init_rms={init_rms:.6f}, final_rms={final_rms:.6f}, delta_rms={delta_rms:.6f}")


if __name__ == "__main__":
    main()
