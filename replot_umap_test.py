#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap


CLASS_ORDER = [1, 2, 4, 5, 6, 8, 13]
CLASS_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    4: "#2ca02c",
    5: "#d62728",
    6: "#9467bd",
    8: "#8c564b",
    13: "#e377c2",
}


DEFAULT_FEATURES_DIR = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/"
    "runs/EXP8_no_group_branch_starembed_features"
)
DEFAULT_RESULT_DIR = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/"
    "runs/EXP8_no_group_branch_starembed_features/benchmark/x/clustering/"
    "EXP8_no_group_branch_starembed_features_all_concat_std0_p30.0_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot the EXP8 test split embeddings with UMAP using the original class labels."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing starembed_embeddings_test.npz.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Original clustering result directory containing label_to_idx.json and text_labels.json.",
    )
    parser.add_argument(
        "--scenario",
        choices=["concat", "avg"],
        default="concat",
        help="concat uses NPZ key 'x'; avg uses NPZ key 'x_avg'.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--font-size", type=float, default=16.0)
    parser.add_argument("--legend-font-size", type=float, default=13.0)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--plot-fraction", type=float, default=0.65)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to <result-dir>/umap_Test_Split.pdf.",
    )
    return parser.parse_args()


def load_original_labels(result_dir: Path) -> tuple[dict[int, int], list[str]]:
    label_to_idx_path = result_dir / "label_to_idx.json"
    text_labels_path = result_dir / "text_labels.json"
    if not label_to_idx_path.exists() or not text_labels_path.exists():
        raise FileNotFoundError(
            "Expected original label files in result_dir: label_to_idx.json and text_labels.json"
        )

    with label_to_idx_path.open() as f:
        label_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with text_labels_path.open() as f:
        text_labels = [str(label).replace("RS_CVn", "RS CVn") for label in json.load(f)]

    return label_to_idx, text_labels


def load_test_split(features_dir: Path, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    npz_path = features_dir / "starembed_embeddings_test.npz"
    key = "x" if scenario == "concat" else "x_avg"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing test split: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {list(data.keys())}")
        if "y" not in data:
            raise KeyError(f"Key 'y' not found in {npz_path}. Available keys: {list(data.keys())}")

        X = np.asarray(data[key], dtype=np.float32)
        y_orig = np.asarray(data["y"], dtype=np.int64)

    finite = np.all(np.isfinite(X), axis=1)
    return X[finite], y_orig[finite]


def remap_labels(y_orig: np.ndarray, label_to_idx: dict[int, int]) -> np.ndarray:
    missing = sorted(set(map(int, y_orig)) - set(label_to_idx))
    if missing:
        raise KeyError(f"Test labels missing from original label_to_idx.json: {missing}")
    return np.asarray([label_to_idx[int(y)] for y in y_orig], dtype=np.int64)


def select_plot_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(labels.shape[0])
    if fraction <= 0.0:
        raise ValueError("--plot-fraction must be positive")

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        n_keep = max(1, int(np.ceil(idx.size * fraction)))
        selected.append(np.sort(rng.choice(idx, size=n_keep, replace=False)))
    return np.sort(np.concatenate(selected))


def plot_umap(
    X2: np.ndarray,
    labels: np.ndarray,
    text_labels: list[str],
    label_to_idx: dict[int, int],
    output: Path,
    *,
    font_size: float,
    legend_font_size: float,
    point_size: float,
    alpha: float,
    plot_fraction: float,
    seed: int,
) -> None:
    plot_idx = select_plot_indices(labels, plot_fraction, seed)
    plot_mask = np.zeros(labels.shape[0], dtype=bool)
    plot_mask[plot_idx] = True

    with plt.rc_context(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": legend_font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.6, 6.5))
        for orig_label in CLASS_ORDER:
            if orig_label not in label_to_idx:
                continue
            label_idx = label_to_idx[orig_label]
            mask = (labels == label_idx) & plot_mask
            if not np.any(mask):
                continue
            label_text = text_labels[label_idx] if 0 <= label_idx < len(text_labels) else str(orig_label)
            ax.scatter(
                X2[mask, 0],
                X2[mask, 1],
                s=point_size,
                label=label_text,
                alpha=alpha,
                color=CLASS_COLORS[orig_label],
                linewidths=0,
            )

        ax.legend(loc="best", frameon=False, markerscale=1.8)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        if output.suffix.lower() == ".pdf":
            fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output or (args.result_dir / "umap_Test_Split.pdf")

    label_to_idx, text_labels = load_original_labels(args.result_dir)
    X_test, y_test_orig = load_test_split(args.features_dir, args.scenario)
    y_test = remap_labels(y_test_orig, label_to_idx)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    X_umap = reducer.fit_transform(X_test)

    np.savez_compressed(output.with_suffix(".npz"), X_umap=X_umap.astype(np.float32), y=y_test)
    plot_umap(
        X_umap,
        y_test,
        text_labels,
        label_to_idx,
        output,
        font_size=float(args.font_size),
        legend_font_size=float(args.legend_font_size),
        point_size=float(args.point_size),
        alpha=float(args.alpha),
        plot_fraction=float(args.plot_fraction),
        seed=int(args.random_state),
    )
    print(f"Saved UMAP test split plot to {output}")


if __name__ == "__main__":
    main()
