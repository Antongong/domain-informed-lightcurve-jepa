#!/usr/bin/env python3
"""Visualize numeric value embeddings as PCA RGB strips for EXP8 and EXP12."""

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

    # Shape is [num_value_bins, embedding_dim].
    embedding = state[weight_key].detach().float().cpu().numpy()

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

    values = np.linspace(vmin, vmax, embedding.shape[0], dtype=np.float32)
    return values, embedding, use_uncertainty


def pca_fit_transform(mats: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate(mats, axis=0).astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]
    scores = [(mat.astype(np.float64) - mean) @ components.T for mat in mats]
    explained = singular_values**2 / max(centered.shape[0] - 1, 1)
    explained_ratio = explained[:3] / explained.sum()
    return scores, components, mean.squeeze(0), explained_ratio


def scores_to_rgb(scores: list[np.ndarray], low_pct: float, high_pct: float) -> list[np.ndarray]:
    stacked = np.concatenate(scores, axis=0)
    lo = np.percentile(stacked, low_pct, axis=0)
    hi = np.percentile(stacked, high_pct, axis=0)
    denom = np.maximum(hi - lo, 1.0e-12)
    rgbs = []
    for score in scores:
        rgb = (score - lo) / denom
        rgbs.append(np.clip(rgb, 0.0, 1.0))
    return rgbs


def plot_rgb_strips(
    records: list[tuple[ExperimentSpec, np.ndarray, np.ndarray, bool]],
    explained_ratio: np.ndarray,
    out_path: Path,
    strip_height: int,
) -> None:
    fig, axes = plt.subplots(len(records), 1, figsize=(13, 3.8), sharex=True)
    if len(records) == 1:
        axes = [axes]

    for ax, (spec, values, rgb, use_uncertainty) in zip(axes, records):
        strip = np.repeat(rgb[np.newaxis, :, :], strip_height, axis=0)
        ax.imshow(
            strip,
            aspect="auto",
            origin="lower",
            extent=[float(values[0]), float(values[-1]), 0.0, 1.0],
            interpolation="nearest",
        )
        suffix = "enabled" if use_uncertainty else "disabled"
        ax.set_yticks([])
        ax.set_ylabel(spec.run_dir.name.replace("_", "\n"), rotation=0, labelpad=70, va="center")
        ax.set_title(f"{spec.label}; error-aware path {suffix}", fontsize=11)

    axes[-1].set_xlabel("True numeric value")
    title = (
        "Raw Numeric Embedding PCA RGB Strip "
        f"(R=PC1 {explained_ratio[0]*100:.1f}%, "
        f"G=PC2 {explained_ratio[1]*100:.1f}%, "
        f"B=PC3 {explained_ratio[2]*100:.1f}%)"
    )
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_scores_csv(
    records: list[tuple[ExperimentSpec, np.ndarray, np.ndarray, bool]],
    score_records: list[np.ndarray],
    out_path: Path,
) -> None:
    rows = []
    for (spec, values, _rgb, _use_uncertainty), scores in zip(records, score_records):
        for value, score in zip(values, scores):
            rows.append(
                {
                    "experiment": spec.run_dir.name,
                    "numeric_value": float(value),
                    "pc1": float(score[0]),
                    "pc2": float(score[1]),
                    "pc3": float(score[2]),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--view", choices=["raw", "phase", "periodogram"], default="raw")
    parser.add_argument("--low_pct", type=float, default=1.0)
    parser.add_argument("--high_pct", type=float, default=99.0)
    parser.add_argument("--strip_height", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    embeddings = []
    for spec in EXPERIMENTS:
        values, embedding, use_uncertainty = load_embedding(spec.run_dir, args.view)
        loaded.append((spec, values, embedding, use_uncertainty))
        embeddings.append(embedding)

    scores, _components, _mean, explained_ratio = pca_fit_transform(embeddings)
    rgbs = scores_to_rgb(scores, args.low_pct, args.high_pct)

    records = [
        (spec, values, rgb, use_uncertainty)
        for (spec, values, _embedding, use_uncertainty), rgb in zip(loaded, rgbs)
    ]

    out_path = args.out_dir / f"EXP8_vs_EXP12_{args.view}_numeric_embedding_pca_rgb_strip.png"
    plot_rgb_strips(records, explained_ratio, out_path, args.strip_height)
    print(f"Wrote {out_path}")

    csv_path = args.out_dir / f"EXP8_vs_EXP12_{args.view}_numeric_embedding_pca_scores.csv"
    write_scores_csv(records, scores, csv_path)
    print(f"Wrote {csv_path}")
    print(
        "Explained variance ratio: "
        f"PC1={explained_ratio[0]:.6f}, PC2={explained_ratio[1]:.6f}, PC3={explained_ratio[2]:.6f}"
    )


if __name__ == "__main__":
    main()
