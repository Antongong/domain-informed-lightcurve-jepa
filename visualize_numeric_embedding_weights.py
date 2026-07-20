#!/usr/bin/env python3
"""Visualize learned numeric value-embedding weights for EXP8 and EXP12."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist


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


def load_embedding_weight(run_dir: Path, view: str) -> tuple[np.ndarray, np.ndarray, bool]:
    cfg = load_yaml(run_dir / "config_used.yaml")
    ckpt_path = run_dir / "ckpt_final.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    view_to_key = {
        "raw": "raw_value_embed.embedding.weight",
        "phase": "phase_value_embed.embedding.weight",
        "periodogram": "period_value_embed.embedding.weight",
    }
    if view not in view_to_key:
        raise ValueError(f"Unsupported view {view!r}; choose from {sorted(view_to_key)}")

    weight_key = view_to_key[view]
    if weight_key not in state:
        raise KeyError(f"{weight_key!r} not found in {ckpt_path}")

    weight = state[weight_key].detach().float().cpu().numpy().T

    numc = get_nested(cfg, ["model", "numeric"], {}) or {}
    if view == "periodogram":
        view_cfg = numc.get("periodogram", {}) or {}
        vmin = float(view_cfg.get("vmin", -6.0))
        vmax = float(view_cfg.get("vmax", 2.0))
        use_uncertainty = False
    else:
        view_cfg = numc.get("raw", {}) or {}
        vmin = float(view_cfg.get("vmin", -2.0))
        vmax = float(view_cfg.get("vmax", 2.0))
        if view == "phase":
            view_cfg = numc.get("phase_folded", {}) or {}
        use_uncertainty = bool(view_cfg.get("use_uncertainty", True))
        ablation_unc = get_nested(cfg, ["model", "ablations", "use_uncertainty"], None)
        if ablation_unc is not None:
            use_uncertainty = bool(ablation_unc)

    values = np.linspace(vmin, vmax, weight.shape[1], dtype=np.float32)
    return values, weight, use_uncertainty


def clustered_row_order(weight: np.ndarray, metric: str) -> np.ndarray:
    """Order rows so adjacent embedding dimensions are close under a distance metric."""
    distances = pdist(weight, metric=metric)
    if not np.all(np.isfinite(distances)):
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)
    tree = linkage(distances, method="average")
    ordered_tree = optimal_leaf_ordering(tree, distances)
    return leaves_list(ordered_tree)


def plot_single(
    values: np.ndarray,
    weight: np.ndarray,
    title: str,
    out_path: Path,
    *,
    cmap: str,
    color_abs_max: float | None = None,
    row_metric: str | None = None,
) -> None:
    vmax = float(color_abs_max if color_abs_max is not None else np.nanmax(np.abs(weight)))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(
        weight,
        aspect="auto",
        origin="lower",
        extent=[float(values[0]), float(values[-1]), 0, weight.shape[0] - 1],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("True numeric value")
    y_label = "Embedding dim"
    if row_metric:
        y_label += f" ({row_metric}-clustered)"
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Embedding weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def robust_color_abs_max(records: list[tuple[ExperimentSpec, np.ndarray, np.ndarray, bool]], percentile: float) -> float:
    values = np.concatenate([np.abs(weight).ravel() for _, _, weight, _ in records])
    vmax = float(np.nanpercentile(values, percentile))
    return max(vmax, 1.0e-6)


def plot_combined(
    records: list[tuple[ExperimentSpec, np.ndarray, np.ndarray, bool]],
    out_path: Path,
    cmap: str,
    color_abs_max: float,
    row_metric: str | None,
) -> None:
    fig, axes = plt.subplots(1, len(records), figsize=(14.5, 5.3), sharey=True)
    if len(records) == 1:
        axes = [axes]

    im = None
    for ax, (spec, values, weight, use_uncertainty) in zip(axes, records):
        im = ax.imshow(
            weight,
            aspect="auto",
            origin="lower",
            extent=[float(values[0]), float(values[-1]), 0, weight.shape[0] - 1],
            cmap=cmap,
            vmin=-color_abs_max,
            vmax=color_abs_max,
            interpolation="nearest",
        )
        suffix = "enabled" if use_uncertainty else "disabled"
        ax.set_title(f"{spec.label}\nerror-aware path {suffix}")
        ax.set_xlabel("True numeric value")
        ax.grid(False)

    y_label = "Embedding dim"
    if row_metric:
        y_label += f" ({row_metric}-clustered)"
    axes[0].set_ylabel(y_label)
    assert im is not None
    cbar = fig.colorbar(im, ax=axes, pad=0.015)
    cbar.set_label("Embedding weight")
    fig.suptitle("Learned Raw Numeric Value-Embedding Weights", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--view", choices=["raw", "phase", "periodogram"], default="raw")
    parser.add_argument("--cmap", default="seismic")
    parser.add_argument(
        "--contrast_percentile",
        type=float,
        default=97.5,
        help="Use this percentile of absolute weights for symmetric color limits.",
    )
    parser.add_argument(
        "--sort_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reorder embedding-dimension rows by distance-based clustering.",
    )
    parser.add_argument(
        "--row_metric",
        default="euclidean",
        help="Distance metric passed to scipy.spatial.distance.pdist for row sorting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    row_orders = {}
    for spec in EXPERIMENTS:
        values, weight, use_uncertainty = load_embedding_weight(spec.run_dir, args.view)
        if args.sort_rows:
            order = clustered_row_order(weight, args.row_metric)
            row_orders[spec.run_dir.name] = order
            weight = weight[order]
        records.append((spec, values, weight, use_uncertainty))

    color_abs_max = robust_color_abs_max(records, args.contrast_percentile)
    for spec, values, weight, use_uncertainty in records:
        slug = spec.run_dir.name
        suffix = "error_aware" if use_uncertainty else "no_error_aware"
        out_path = args.out_dir / f"{slug}_{args.view}_numeric_embedding_weight_{suffix}.png"
        plot_single(
            values,
            weight,
            f"{spec.label} ({args.view}); error-aware path {'enabled' if use_uncertainty else 'disabled'}"
            + (f"; rows {args.row_metric}-clustered" if args.sort_rows else ""),
            out_path,
            cmap=args.cmap,
            color_abs_max=color_abs_max,
            row_metric=args.row_metric if args.sort_rows else None,
        )
        print(f"Wrote {out_path}")

    combined_path = args.out_dir / f"EXP8_vs_EXP12_{args.view}_numeric_embedding_weights.png"
    plot_combined(records, combined_path, args.cmap, color_abs_max, args.row_metric if args.sort_rows else None)
    print(f"Wrote {combined_path}")
    print(f"Color limits: +/-{color_abs_max:.4g} ({args.contrast_percentile:g}th percentile of abs weights)")
    for slug, order in row_orders.items():
        metric_slug = args.row_metric.replace(" ", "_")
        order_path = args.out_dir / f"{slug}_{args.view}_{metric_slug}_row_order.npy"
        np.save(order_path, order)
        print(f"Wrote {order_path}")


if __name__ == "__main__":
    main()
